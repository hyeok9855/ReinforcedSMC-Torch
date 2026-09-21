import hashlib
import json
from pathlib import Path

import numpy as np
import torch

REFERENCE_DIR = Path(__file__).parent.parent / "energies" / "data" / "references"

# (energy_name, ndim) -> directory name used by 1step_energy_sampler
TARGET_DIRS = {
    ("funnel", 10): "FunnelOracle_d10",
    ("manywell", 32): "ManyWellOracle_d32",
    ("manywell", 64): "ManyWellOracle_d64",
    ("gmm40", 2): "MoGOracle_d2",
    ("gmm40", 10): "MoGOracle_d10",
    ("nice_mnist", 196): "nice_mnist_d196",
    ("nice_fashion_mnist", 784): "nice_fashion_mnist_d784",
}

_LFS_MAGIC = b"version https://git-lfs.github.com/spec/v1"


def target_dir(energy_name: str, ndim: int) -> Path:
    key = (energy_name, ndim)
    if key not in TARGET_DIRS:
        raise KeyError(
            f"No pinned reference for {energy_name} at ndim={ndim}. "
            f"Available: {sorted(TARGET_DIRS)}"
        )
    return REFERENCE_DIR / TARGET_DIRS[key]


def reference_bandwidth(energy_name: str, ndim: int) -> float:
    """Fixed RBF bandwidth for `reference_mmd_squared` on this target."""
    with open(target_dir(energy_name, ndim) / "bandwidth.json") as f:
        return float(json.load(f)["bandwidth"])


def noise_floor(energy_name: str, ndim: int, seed: int = 0) -> dict:
    """Metrics of ground_truth vs reference -- the best score achievable at n=10_000."""
    with open(target_dir(energy_name, ndim) / f"seed_{seed}" / "complete.json") as f:
        return json.load(f)["metrics"]


def load_reference(
    energy_name: str,
    ndim: int,
    seed: int = 0,
    which: str = "reference",
    device: str | torch.device = "cpu",
    verify: bool = True,
) -> torch.Tensor:
    """Load a pinned 10_000-sample evaluation set as float32 [10000, ndim].

    `which` is "reference" (score model samples against this) or "ground_truth"
    (the second draw, used only to reproduce the noise floor).
    """
    if which not in ("reference", "ground_truth"):
        raise ValueError(f"which must be 'reference' or 'ground_truth' (got {which!r})")

    tdir = target_dir(energy_name, ndim)
    path = tdir / f"seed_{seed}" / f"{which}.npy"
    with open(tdir / f"seed_{seed}" / "complete.json") as f:
        expected = json.load(f)[which]["sha256"]

    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing. The pinned evaluation samples are git-lfs objects in a "
            f"private repo and are not committed here; obtain the file (expected sha256 "
            f"{expected}) and drop it at that path."
        )
    raw = path.read_bytes()
    if raw.startswith(_LFS_MAGIC):
        raise FileNotFoundError(
            f"{path} is a git-lfs pointer stub, not the array. Run `git lfs pull` in the "
            f"source repo and copy the real file here (expected sha256 {expected})."
        )
    if verify:
        got = hashlib.sha256(raw).hexdigest()
        if got != expected:
            raise ValueError(
                f"{path} sha256 mismatch: expected {expected[:16]}…, got {got[:16]}…. "
                "This is not the pinned evaluation set; metrics would not be comparable."
            )

    arr = np.load(path)
    if arr.ndim != 2 or arr.shape[1] != ndim:
        raise ValueError(f"{path}: expected [N, {ndim}], got {tuple(arr.shape)}")
    return torch.from_numpy(arr).to(dtype=torch.float32, device=device)
