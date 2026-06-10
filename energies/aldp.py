import typing
import math
import warnings
from pathlib import Path

import boltzgen as bg
import matplotlib.pyplot as plt
import mdtraj as md
import numpy as np
import torch
import wandb
from matplotlib.colors import LogNorm
from openmmtools import testsystems

from energies.base import BaseEnergy
from utils.misc_utils import temp_seed
from utils.plot_utils import fig_to_image, viz_interatomic_dist_hist, viz_energy_hist


DATA_PATH = Path(__file__).parent / "data" / "aldp"
PI_PLUS_EPS = math.pi + 0.0001


class ALDP(BaseEnergy):
    is_particle_system = True

    def __init__(
        self,
        device: str | torch.device = "cpu",
        temperature=300,
        energy_cut=1.0e5,  # 1e8 in FAB
        energy_max=1.0e20,  # 1e20 in FAB
        n_threads=16,
        ind_circ_dih=[0, 1, 2, 3, 4, 5, 8, 9, 10, 13, 15, 16],
        shift_dih=False,
        shift_dih_params={"hist_bins": 100},
        default_std={"bond": 0.005, "angle": 0.15, "dih": 0.2},
        env="implicit",
        seed: int = 0,
        chirality_ind=[17, 26],
        chirality_mean_diff=-0.043,
        chirality_threshold=0.8,
        chirality_sharpness=100.0,
    ):
        super().__init__(device=device, ndim=60, seed=seed)  # 60 since we use internal coordinates

        # Define molecule parameters
        z_matrix = [
            (0, [1, 4, 6]),
            (1, [4, 6, 8]),
            (2, [1, 4, 0]),
            (3, [1, 4, 0]),
            (4, [6, 8, 14]),
            (5, [4, 6, 8]),
            (7, [6, 8, 4]),
            (9, [8, 6, 4]),
            (10, [8, 6, 4]),
            (11, [10, 8, 6]),
            (12, [10, 8, 11]),
            (13, [10, 8, 11]),
            (15, [14, 8, 16]),
            (16, [14, 8, 6]),
            (17, [16, 14, 15]),
            (18, [16, 14, 8]),
            (19, [18, 16, 14]),
            (20, [18, 16, 19]),
            (21, [18, 16, 19]),
        ]

        mass = [
            [1.007947, 1.007947, 1.007947],
            [12.01078, 12.01078, 12.01078],
            [1.007947, 1.007947, 1.007947],
            [1.007947, 1.007947, 1.007947],
            [12.01078, 12.01078, 12.01078],
            [15.99943, 15.99943, 15.99943],
            [14.00672, 14.00672, 14.00672],
            [1.007947, 1.007947, 1.007947],
            [12.01078, 12.01078, 12.01078],
            [1.007947, 1.007947, 1.007947],
            [12.01078, 12.01078, 12.01078],
            [1.007947, 1.007947, 1.007947],
            [1.007947, 1.007947, 1.007947],
            [1.007947, 1.007947, 1.007947],
            [12.01078, 12.01078, 12.01078],
            [15.99943, 15.99943, 15.99943],
            [14.00672, 14.00672, 14.00672],
            [1.007947, 1.007947, 1.007947],
            [12.01078, 12.01078, 12.01078],
            [1.007947, 1.007947, 1.007947],
            [1.007947, 1.007947, 1.007947],
            [1.007947, 1.007947, 1.007947],
        ]
        self.mass = torch.tensor(mass, device=self.device).unsqueeze(0)
        self.kBT = 1.380649 * 6.02214076 * 1e-3 * temperature
        self.beta = 1 / self.kBT

        cart_indices = [8, 6, 14]

        # System setup
        if env == "vacuum":
            system = testsystems.AlanineDipeptideVacuum(constraints=None)
        elif env == "implicit":
            system = testsystems.AlanineDipeptideImplicit(constraints=None)
        else:
            raise NotImplementedError("This environment is not implemented.")

        dtype = torch.get_default_dtype()
        transform_data = torch.load(DATA_PATH / "position_min_energy.pt", weights_only=True)
        transform_data = transform_data.to(dtype)

        # Set distribution
        self.coordinate_transform = bg.flows.CoordinateTransform(
            transform_data,
            66,  # 66 is after transform
            z_matrix,
            cart_indices,
            mode="internal",
            ind_circ_dih=ind_circ_dih,
            shift_dih=shift_dih,
            shift_dih_params=shift_dih_params,
            default_std=default_std,
        )
        self.coordinate_transform.to(self.device)

        self.energy_cut = energy_cut

        self.p = bg.distributions.TransformedBoltzmannParallel(
            system,
            temperature,
            energy_cut=energy_cut,
            energy_max=energy_max,
            transform=self.coordinate_transform,
            n_threads=n_threads,
        )
        self.p.to(self.device)

        ncarts = self.coordinate_transform.transform.len_cart_inds
        permute_inv = self.coordinate_transform.transform.permute_inv.cpu().numpy()
        dih_ind = self.coordinate_transform.transform.ic_transform.dih_indices.cpu().numpy()

        ind = torch.arange(self.ndim)
        ind = torch.cat(
            [ind[: 3 * ncarts - 6], -torch.ones(6, dtype=torch.int64), ind[3 * ncarts - 6 :]]
        )
        ind = ind[permute_inv]
        dih_ind = ind[dih_ind]
        self.ind_circ = dih_ind[ind_circ_dih].tolist()

        self.chirality_ind = chirality_ind
        self.chirality_mean_diff = chirality_mean_diff
        self.chirality_threshold = chirality_threshold
        self.chirality_sharpness = chirality_sharpness

        datas = [np.load(DATA_PATH / f"val_before_scale{i}.npy") for i in range(5)]
        approx_sample = torch.tensor(np.concatenate(datas, axis=0), dtype=dtype)
        self.approx_sample = approx_sample[self.get_lform_indices(approx_sample)]

    def energy(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[1] == 60
        x_fab, log_det = self.scale_ind_circ(x)
        energy = -(self.p.log_prob(x_fab) + log_det) + self._compute_chirality_penalty(x)
        energy[energy.isnan()] = 2 * self.energy_cut
        return energy

    def get_lform_indices(self, x: torch.Tensor) -> torch.Tensor:
        # Compute the dihedral angle difference
        diff_ = torch.column_stack(
            (
                x[:, self.chirality_ind[0]] - x[:, self.chirality_ind[1]],
                x[:, self.chirality_ind[0]] - x[:, self.chirality_ind[1]] + 2 * np.pi,
                x[:, self.chirality_ind[0]] - x[:, self.chirality_ind[1]] - 2 * np.pi,
            )
        )

        # Find the minimal angular difference (handling periodicity)
        min_diff_ind = torch.min(torch.abs(diff_), dim=1).indices
        diff = diff_[torch.arange(x.shape[0]), min_diff_ind]

        # Compute deviation from the L-form reference
        deviation = torch.abs(diff - self.chirality_mean_diff)

        # High energy for non-L-form
        is_l_form = deviation < self.chirality_threshold
        return is_l_form

    def _compute_chirality_penalty(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute smooth chirality penalty that encourages L-form configurations.

        Args:
            x: Input tensor of shape (batch_size, 60) in the internal coordinate space

        Returns:
            Penalty energy for each sample in the batch
        """
        is_l_form = self.get_lform_indices(x)
        penalty = (2 * self.energy_cut) * (1 - is_l_form.float())

        # # Smooth penalty using sigmoid function
        # # - When deviation < threshold: penalty ≈ 0 (L-form region)
        # # - When deviation > threshold: penalty increases smoothly (D-form region)
        # penalty = self.chirality_penalty_strength * torch.sigmoid(
        #     (deviation - self.chirality_threshold) * self.chirality_sharpness
        # )

        return penalty

    def sample(self, batch_size: int, seed: int | None = None) -> torch.Tensor:
        with temp_seed(seed or self.seed):
            perm_idx = torch.randperm(self.approx_sample.shape[0])[:batch_size]
        return self.approx_sample[perm_idx].to(self.device)

    def scale_ind_circ(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        assert x.shape[1] == 60
        x_fab = x.clone()
        x_fab[:, self.ind_circ] = torch.tanh(x_fab[:, self.ind_circ]) * PI_PLUS_EPS
        log_cosh_x_ind_circ = torch.log(torch.cosh(x[:, self.ind_circ]))
        log_det = -2 * log_cosh_x_ind_circ + math.log(PI_PLUS_EPS)
        return x_fab, log_det.sum(1)

    def inverse_scale_ind_circ(self, x_fab: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        assert x_fab.shape[1] == 60
        x = x_fab.clone()
        x[:, self.ind_circ] = torch.atanh(x[:, self.ind_circ] / PI_PLUS_EPS)
        log_det = math.log(PI_PLUS_EPS) - torch.log(
            (PI_PLUS_EPS**2 - x_fab[:, self.ind_circ] ** 2).abs()
        )
        return x, log_det.sum(1)

    ### Some methods for MD ###
    def transform(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # scale_ind_circ --> (60 -> 66)
        assert x.shape[1] == 60
        x_fab, log_det_fab = self.scale_ind_circ(x)
        x_orig, log_det_orig = self.coordinate_transform(x_fab)
        return x_orig, log_det_fab + log_det_orig

    def inverse(self, x_orig: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # (66 -> 60) --> inverse_scale_ind_circ
        assert x_orig.shape[1] == 66
        x_fab, log_det_fab = self.coordinate_transform.inverse(x_orig)
        x, log_det_x = self.inverse_scale_ind_circ(x_fab)
        return x, log_det_fab + log_det_x

    ### Visualization ###
    def visualize(
        self, samples: torch.Tensor, weights: torch.Tensor | None = None, **kwargs
    ) -> dict:
        if weights is not None:
            warnings.warn("Can't visualize weighted samples for ALDP. Ignoring them...")
            return {}
        out_dict = {}
        out_dict.update(self.plot_phi_psi(samples))
        out_dict.update(self.draw_mols(samples))
        out_dict.update(viz_energy_hist(self, samples))
        out_dict.update(viz_interatomic_dist_hist(self, samples))
        return out_dict

    def plot_phi_psi(self, xs: torch.Tensor, dpi=300):
        """
        Plots a 2D histogram of phi and psi angles.

        Args:
            xs (torch.Tensor): Input data for dihedral angle computation.
            dpi (int): Dots per inch for the figure.

        Returns:
            matplotlib.figure.Figure: The generated figure.
        """
        xs, _ = self.transform(xs)

        assert xs.ndim == 2  # (n_samples, n_atoms * 3)
        xs = xs.reshape(xs.shape[0], -1, 3)  # (n_samples, n_atoms, 3)

        def compute_dihedral(positions: torch.Tensor) -> torch.Tensor:
            v = positions[:, :-1] - positions[:, 1:]
            v0 = -v[:, 0]
            v1 = v[:, 2]
            v2 = v[:, 1]

            s0 = torch.sum(v0 * v2, dim=-1, keepdim=True) / torch.sum(v2 * v2, dim=-1, keepdim=True)
            s1 = torch.sum(v1 * v2, dim=-1, keepdim=True) / torch.sum(v2 * v2, dim=-1, keepdim=True)

            v0 = v0 - s0 * v2
            v1 = v1 - s1 * v2

            v0 = v0 / torch.norm(v0, dim=-1, keepdim=True)
            v1 = v1 / torch.norm(v1, dim=-1, keepdim=True)
            v2 = v2 / torch.norm(v2, dim=-1, keepdim=True)

            x = torch.sum(v0 * v1, dim=-1)
            v3 = torch.cross(v0, v2, dim=-1)
            y = torch.sum(v3 * v1, dim=-1)
            return torch.atan2(y, x)

        fig = plt.figure(figsize=(7, 7), dpi=dpi)

        angle_1 = [6, 8, 14, 16]
        angle_2 = [1, 6, 8, 14]

        psi = compute_dihedral(xs[:, angle_1, :])
        phi = compute_dihedral(xs[:, angle_2, :])
        phi = phi.detach().cpu().numpy()
        psi = psi.detach().cpu().numpy()

        xedges = np.linspace(-np.pi, np.pi, 51)
        yedges = np.linspace(-np.pi, np.pi, 51)
        plt.hist2d(phi, psi, bins=[xedges, yedges], norm=LogNorm(), cmap="viridis")
        plt.xlim(-np.pi, np.pi)
        plt.ylim(-np.pi, np.pi)
        plt.xlabel("$\phi$", fontsize=28)
        plt.ylabel("$\psi$", fontsize=28)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.xticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi], ["-π", "-π/2", "0", "π/2", "π"])
        plt.yticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi], ["-π", "-π/2", "0", "π/2", "π"])
        # plt.colorbar(label="Density")
        plt.tight_layout()
        return {"visualization/phi_psi": wandb.Image(fig_to_image(fig))}

    def draw_mols(self, xs: torch.Tensor, name="aldp"):
        """
        Draw a figure containing 3D molecule.

        Args:
            energy (BaseEnergy): Energy function.
            sample (Array): Sample generated by model.

        Return:
            fig, axs: matplotlib figure and axes objec
        """
        xs, _ = self.transform(xs)

        assert xs.shape[0] >= 3

        # Make ten subplots
        fig, axs = plt.subplots(1, 3, figsize=(30, 10), subplot_kw=dict(projection="3d"))

        for i, ax in enumerate(axs.flatten()):
            draw_mol(
                name,
                ax,
                xs[i].reshape(-1, 3).detach().cpu().numpy(),
            )
        return {"visualization/3D": wandb.Image(fig_to_image(fig))}


### Helper functions for visualization ###

ATOM_COLORS = {
    "carbon": "gray",
    "nitrogen": "blue",
    "oxygen": "red",
    "hydrogen": "black",
    "sulfur": "yellow",
    "phosphorus": "orange",
}

ATOM_SIZES = {
    "carbon": 100,
    "nitrogen": 100,
    "oxygen": 100,
    "hydrogen": 25,
    "sulfur": 100,
    "phosphorus": 100,
}


@typing.no_type_check
def draw_mol(name: str, ax: plt.Axes, coordinate: np.ndarray) -> plt.Axes:
    """
    Visualizes molecular conformation using matplotlib's 3D plot.
    Returns the generated matplotlib Axes object.

    parameters:
        coordinates (Array): Molecular atom coordinates. Should be array of shape (n_atoms, 3).

    return:
        matplotlib.axes.Axes: Axes object containing the visualized molecular plot.
    """

    # get topology (md.Topology) from pdb file
    coordinate = np.nan_to_num(coordinate, nan=0.0, posinf=0.0, neginf=0.0)

    topology = md.load(DATA_PATH / f"{name}.pdb").topology

    center_of_mass = np.mean(coordinate, axis=0)
    coordinate = coordinate - center_of_mass

    # Set the box aspect ratio
    ax.set_aspect("equal")

    # Set the background color to white
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor("w")
    ax.yaxis.pane.set_edgecolor("w")
    ax.zaxis.pane.set_edgecolor("w")

    # Draw atoms
    for i, atom in enumerate(topology.atoms):
        atom_name = atom.element.name
        ax.scatter(
            coordinate[i, 0],
            coordinate[i, 1],
            coordinate[i, 2],
            c=ATOM_COLORS.get(atom_name, "gray"),
            s=ATOM_SIZES.get(atom_name, 20),
            label=atom_name,
            alpha=0.8,
            edgecolors="black",
            depthshade=True,
        )

    # Draw bonds
    for bond in topology.bonds:
        atom1, atom2 = bond
        x = [coordinate[atom1.index, 0], coordinate[atom2.index, 0]]
        y = [coordinate[atom1.index, 1], coordinate[atom2.index, 1]]
        z = [coordinate[atom1.index, 2], coordinate[atom2.index, 2]]
        ax.plot(x, y, z, "k-", linewidth=2.0, alpha=0.6)

    # Set the view angle
    ax.view_init(elev=20, azim=45)

    # Set the axis labels
    ax.set_xlabel("X (nm)")
    ax.set_ylabel("Y (nm)")
    ax.set_zlabel("Z (nm)")

    # Adjust the axis limits to fit the molecule
    max_range = (
        np.array(
            [
                coordinate[:, 0].max() - coordinate[:, 0].min(),
                coordinate[:, 1].max() - coordinate[:, 1].min(),
                coordinate[:, 2].max() - coordinate[:, 2].min(),
            ]
        ).max()
        / 2.0
    )
    mid_x = (coordinate[:, 0].max() + coordinate[:, 0].min()) * 0.5
    mid_y = (coordinate[:, 1].max() + coordinate[:, 1].min()) * 0.5
    mid_z = (coordinate[:, 2].max() + coordinate[:, 2].min()) * 0.5

    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    # Draw the legend
    handles, labels = ax.get_legend_handles_labels()
    unique_labels = dict(zip(labels, handles))
    ax.legend(
        unique_labels.values(),
        unique_labels.keys(),
        loc="center left",
        bbox_to_anchor=(1, 0.5),
    )

    return ax
