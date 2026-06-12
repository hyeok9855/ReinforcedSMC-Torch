# Reinforced Sequential Monte Carlo for Amortised Sampling (PyTorch Version)

This repository contains the PyTorch implementation of the paper "[Reinforced Sequential Monte Carlo for Amortised Sampling](https://arxiv.org/abs/2510.11711)" (ICML 2026 spotlight). This repository is especially for the alanine dipeptide (ALDP) conformer sampling experiment (Appendix H). For the main results with synthetic targets, please check the [JAX version](https://github.com/hyeok9855/ReinforcedSMC).

## Installation

We recommend using [uv](https://github.com/astral-sh/uv) to install dependencies and run the project.

First, synchronize the dependencies to set up a virtual environment and install all packages:
```bash
uv sync
```

By default, `uv sync` installs PyTorch built for **CUDA 12.8** (`cu128`). To use a different PyTorch/CUDA build, pass a dependency group instead:

```bash
# CUDA 13.0
uv sync --no-default-groups --group cu130

# CPU-only (e.g. macOS or machines without a GPU)
uv sync --no-default-groups --group cpu
```

This will automatically create a virtual environment (`.venv`) and install all required packages (including PyTorch, OpenMM, openmmtools, and boltzgen).


## Usage (ALDP)

Check scripts/aldp.sh for the detailed settings.

```bash
bash scripts/aldp.sh <ALG> <SEED>
```

<ALG> can be one of the following:
- `pis` (Path Integral Sampler with reverse-KL loss)
- `tb` (On-policy training with Trajectory Balance (TB) loss)
- `tb_buf` (TB with unweighted buffer)
- `tb_iwbuf` (TB with importance-weighted buffer)
- `tb_buf_mcmc` (`tb_buf` with MCMC buffer augmentation)
- `tb_iwbuf_mcmc` (`tb_iwbuf` with MCMC buffer augmentation)
- `tb-subtb_buf_smc` (TB/SubTB combined loss with unweighted buffer and SMC)
- `tb-subtb_iwbuf_smc` (TB/SubTB combined loss with importance-weighted buffer and SMC)
- `tb-subtb_buf_smc_mcmc` (`tb-subtb_buf_smc` with MCMC buffer augmentation)
- `tb-subtb_iwbuf_smc_mcmc` (`tb-subtb_iwbuf_smc` with MCMC buffer augmentation)


## Usage (Synthetic Energy)

**CAVEAT**: This PyTorch version has not been tested for synthetic targets. Please refer to the [JAX-based implementation](https://github.com/hyeok9855/ReinforcedSMC) for the main results with synthetic targets in our paper. Still, you can try it out with this PyTorch version, but the results can be slightly different from the JAX version.

An example usage with gmm40 (5d) energy

```bash
python --energy_name gmm40 --ndim 5 --reference_process ou --init_std 20.0 --num_steps 64 --loss_type tb-subtb --prioritization iw --buffer_target_ess 0.05 --smc --smc_target_ess 0.05 --epochs 20000 --seed 0
```

For different energies, you might need to change the `init_std` accordingly so that the reference process reasonably covers the target space that you want to sample from.


## Citation

If you use this repository in your work, please cite it using the following BibTeX entry:

```bibtex
@inproceedings{choi2026reinforced,
  title={Reinforced Sequential {M}onte {C}arlo for Amortised Sampling},
  author={Choi, Sanghyeok and Mittal, Sarthak and Elvira, V{\'\i}ctor and Park, Jinkyoo and Whitammer, Esmeralda S.},
  booktitle={Forty-third International Conference on Machine Learning},
  year={2026}
}
```
