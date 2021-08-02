import math

import pandas as pd
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import json
import pathlib
import pdffitx.modeling as md
import pdffitx.io as io
import typing as tp
from utils import MultiPhaseModel
import matplotlib.gridspec as gridspec


def ligand(x, amplitube, sigma, phi, wavelength):
    """A simulated PDF for ligands."""
    return amplitube * np.exp(- x ** 2 / (2 * sigma ** 2)) * np.cos(2 * np.pi * x / wavelength + phi)


def shape_b(x, B_psize, B_psig):
    return md.F.lognormalSphericalCF(x, B_psize, B_psig)


def shape_a(x, A_psize):
    return md.F.sphericalCF(x, A_psize)


def create_model_no_ligand():
    bronze = io.load_crystal("stru/TiO2_Bronze_mp.cif")
    anatase = io.load_crystal("stru/TiO2_Anatase.cif")
    model = MultiPhaseModel(
        "fb * B + fa * A",
        {"A": anatase, "B": bronze},
        {"fa": shape_a, "fb": shape_b}
    )
    model.set_param(B_psize=100, B_psig=20, A_psize=50)
    return model


def create_model_with_ligand():
    bronze = io.load_crystal("stru/TiO2_Bronze_mp.cif")
    anatase = io.load_crystal("stru/TiO2_Anatase.cif")
    model = MultiPhaseModel(
        "fb * B + fa * A + L",
        {"A": anatase, "B": bronze},
        {"fa": shape_a, "fb": shape_b, "L": ligand}
    )
    model.set_param(B_psize=100, B_psig=20, A_psize=50, wavelength=5., sigma=10.)
    return model


def load_and_concat(pattern: str, dim: str = "dim_0") -> xr.Dataset:
    dss = []
    names = []
    for f in sorted(pathlib.Path("results").glob(pattern)):
        ds = xr.load_dataset(f)
        dss.append(ds)
        names.append(f.stem)
    ds: xr.Dataset = xr.concat(dss, dim=dim)
    ds = ds.assign({"samples": ("dim_0", names)})
    sample_data = pd.read_csv("./sample_data.csv")
    ds = ds.assign({k: (["dim_0"], list(v)) for k, v in sample_data.to_dict("list").items()})
    return ds


def plot_results(result: xr.Dataset, num_col: int = 4, figure_config: dict = None, grid_config: dict = None, plot_config: dict = None) -> tp.List[plt.Axes]:
    if plot_config is None:
        plot_config = {}
    plot_config.setdefault("marker", "o")
    plot_config.setdefault("ls", "none")
    if figure_config is None:
        figure_config = {}
    if grid_config is None:
        grid_config = {}
    num_row = math.ceil(len(result) / num_col)
    fig: plt.Figure = plt.figure(**figure_config)
    grids = gridspec.GridSpec(num_row, num_col, figure=fig, **grid_config)
    axes = []
    for name, grid in zip(result, grids):
        ax = fig.add_subplot(grid)
        result[name].plot(ax=ax, **plot_config)
        axes.append(ax)
    return axes


def get_phase(result: xr.Dataset, prefix: str) -> xr.Dataset:
    b_names = [name for name in result if name.startswith(prefix)]
    return result[b_names]
