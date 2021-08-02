import math
import string
import typing
import re

import pandas as pd
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import json
import pathlib
import pdffitx.model as md
import pdffitx.io as io
import typing as tp
import matplotlib.gridspec as gridspec
import pdfstream.utils.jupyter as jpt
import pdfstream.visualization.main as vis
import diffpy.srfit.pdf.characteristicfunctions as F

exporter = jpt.FigExporter("./figure", dpi=400)

paper_exporter = jpt.FigExporter("/Users/sst/project/papers/19st_tio2b/paper/figures", dpi=400)


def get_phase(result: xr.Dataset, prefix: str) -> xr.Dataset:
    b_names = [name for name in result if name.startswith(prefix)]
    return result[b_names]


def use_bg_mpl_style() -> None:
    try:
        plt.style.use("/Users/sst/PycharmProjects/bg-mpl-stylesheets/bg_mpl_stylesheet/bg_mpl_stylesheet")
    except FileNotFoundError as e:
        print(e)
    return


def process_param(params: xr.Dataset) -> xr.Dataset:
    s = params["B_scale"] + params["A_scale"]
    params = params.assign(
        {"B_fraction": params["B_scale"] / s * 100, "A_fraction": params["A_scale"] / s * 100}
    )
    params["B_fraction"].attrs.update({"standard_name": "fraction", "units": r"\%"})
    params["A_fraction"].attrs.update({"standard_name": "fraction", "units": r"\%"})
    params["temperature"].attrs.update({"units": r"$^\circ$C"})
    params["time"].attrs.update({"units": "min"})
    params["B_psize"].attrs.update({"long_name": "diameter (PDF)"})
    params["B_psig"].attrs.update({"long_name": "diameter (variance)"})
    params["A_psize"].attrs.update({"long_name": "diameter (PDF)"})
    params["diameter"].attrs.update({"standard_name": "diameter (TEM)", "units": r"Å"})
    return params


def ligand(x, amplitude, sigma, phi, wavelength):
    """A simulated PDF for ligands."""
    return amplitude * np.exp(- x ** 2 / (2 * sigma ** 2)) * np.cos(2 * np.pi * x / wavelength + phi)


def create_model_single_spherical():
    bronze = io.load_crystal("stru/TiO2_Bronze_mp.cif")
    model = md.MultiPhaseModel(
        "B_f * B",
        {"B": bronze},
        {"B_f": F.lognormalSphericalCF}
    )
    model.set_value(B_f_psize=100, B_f_psig=10)
    return model


def create_model_single_bronze():
    bronze = io.load_crystal("stru/TiO2_Bronze_mp.cif")
    model = md.MultiPhaseModel(
        "B_f * B",
        {"B": bronze},
        {"B_f": F.sphericalCF}
    )
    model.set_value(B_f_psize=100)
    return model


def create_model_no_ligand():
    bronze = io.load_crystal("stru/TiO2_Bronze_mp.cif")
    anatase = io.load_crystal("stru/TiO2_Anatase.cif")
    model = md.MultiPhaseModel(
        "B_f * B + A_f * A",
        {"A": anatase, "B": bronze},
        {"A_f": F.sphericalCF, "B_f": F.lognormalSphericalCF}
    )
    model.set_value(B_f_psize=100, B_f_psig=20, A_f_psize=50)
    return model


def create_model_with_ligand():
    bronze = io.load_crystal("stru/TiO2_Bronze_mp.cif")
    anatase = io.load_crystal("stru/TiO2_Anatase.cif")
    model = md.MultiPhaseModel(
        "B_f * B + A_f * A + L",
        {"A": anatase, "B": bronze},
        {"A_f": F.sphericalCF, "B_f": F.lognormalSphericalCF, "L": ligand}
    )
    model.set_value(B_f_psize=100, B_f_psig=20, A_f_psize=50, L_wavelength=5., L_sigma=10.)
    return model


def create_model_spherical():
    bronze = io.load_crystal("stru/TiO2_Bronze_mp.cif")
    anatase = io.load_crystal("stru/TiO2_Anatase.cif")
    model = md.MultiPhaseModel(
        "B_f * B + A_f * A + L",
        {"A": anatase, "B": bronze},
        {"A_f": F.sphericalCF, "B_f": F.sphericalCF, "L": ligand}
    )
    model.set_value(B_f_psize=100, A_f_psize=50, L_wavelength=5., L_sigma=10.)
    return model


def create_model_core_shell():
    bronze = io.load_crystal("stru/TiO2_Bronze_mp.cif")
    anatase = io.load_crystal("stru/TiO2_Anatase.cif")
    model = md.MultiPhaseModel(
        "B_f * B + A_f * A + L",
        {"A": anatase, "B": bronze},
        {"A_f": F.shellCF, "B_f": F.sphericalCF, "L": ligand}
    )
    recipe = model.get_recipe()
    recipe.constrain("B_f_psize", "2 * A_f_radius")
    model.set_value(A_f_radius=100, A_f_thickness=50, L_wavelength=5., L_sigma=10.)
    return model


def bound_strictly(model: md.MultiPhaseModel):
    for name in ["B_a", "B_b", "B_c", "B_beta"]:
        p = model.get_param(name)
        p.boundWindow(0.005)
    for name in ["B_f_psize", "B_f_psig", "A_f_psize"]:
        p = model.get_param(name)
        p.boundWindow(2)
    for name in ["B_Ti0_Biso", "B_Ti1_Biso", "B_O2_Biso", "B_O3_Biso", "B_O4_Biso", "B_O5_Biso", "A_Ti1_Biso", "A_O1_Biso"]:
        p = model.get_param(name)
        p.boundWindow(0.2)


def load_and_concat(directory: str, pattern: str, dim: str = "dim_0") -> xr.Dataset:
    dss = []
    names = []
    sample_data = pd.read_csv("./sample_data.csv", index_col=0)
    fs = []
    for f in pathlib.Path(directory).glob(pattern):
        ds = xr.load_dataset(f)
        sample = f.stem.split("_")[0]
        names.append(sample)
        sr = sample_data.loc[sample]
        ds = ds.assign(dict(zip(sr.index, sr)))
        dss.append(ds)
        fs.append(f.stem)
    ds: xr.Dataset = xr.concat(dss, dim=dim)
    ds = ds.assign({"samples": ("dim_0", names), "files": ("dim_0", fs)})
    ds["temperature"].attrs.update({"units": r"$^\circ$C"})
    ds["time"].attrs.update({"units": "min"})
    return ds


def get_fraction_and_add_units(params: xr.Dataset) -> xr.Dataset:
    s = params["B_scale"] + params["A_scale"]
    params = params.assign(
        {"B_fraction": params["B_scale"] / s * 100, "A_fraction": params["A_scale"] / s * 100}
    )
    params["B_fraction"].attrs.update({"standard_name": "fraction", "units": r"\%"})
    params["A_fraction"].attrs.update({"standard_name": "fraction", "units": r"\%"})
    params["B_f_psize"].attrs.update({"long_name": "diameter (PDF)"})
    params["B_f_psig"].attrs.update({"long_name": "diameter (variance)"})
    params["A_f_psize"].attrs.update({"long_name": "diameter (PDF)"})
    params["diameter"].attrs.update({"standard_name": "diameter (TEM)", "units": r"Å"})
    return params


def process_param2(params: xr.Dataset) -> xr.Dataset:
    s = params["B_scale"] + params["A_scale"]
    params = params.assign(
        {"B_fraction": params["B_scale"] / s * 100, "A_fraction": params["A_scale"] / s * 100}
    )
    params["B_fraction"].attrs.update({"standard_name": "fraction", "units": r"\%"})
    params["A_fraction"].attrs.update({"standard_name": "fraction", "units": r"\%"})
    params["temperature"].attrs.update({"units": r"$^\circ$C"})
    params["time"].attrs.update({"units": "min"})
    params["B_f_psize"].attrs.update({"long_name": "diameter (PDF)"})
    params["B_f_psig"].attrs.update({"long_name": "diameter (variance)"})
    params["A_f_psize"].attrs.update({"long_name": "diameter (PDF)"})
    params["diameter"].attrs.update({"standard_name": "diameter (TEM)", "units": r"Å"})
    return params


def use_parentheses(ax: plt.Axes = None) -> None:
    if ax is None:
        ax = plt.gca()
    xlabel: str = ax.get_xlabel()
    xlabel = xlabel.replace("[", "(").replace("]", ")")
    ax.set_xlabel(xlabel)
    ylabel: str = ax.get_ylabel()
    ylabel = ylabel.replace("[", "(").replace("]", ")")
    ax.set_ylabel(ylabel)
    return


def get_colors() -> list:
    return plt.rcParams['axes.prop_cycle'].by_key()['color']


def replace_cif_reference(refs: typing.Iterable[str], new_sub_str: str) -> typing.List:
    """Replace the "cif-reference-0" to bec new names with increasing index."""
    ans = []
    source = r"@article{.*,"
    for i, ref in enumerate(refs):
        ref2 = re.sub(source, "@article{{{}{},".format(new_sub_str, i), ref)
        ans.append(ref2)
    return ans
