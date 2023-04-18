from typing import Tuple, Dict, Any

import scanpy.preprocessing
from anndata import AnnData
from matplotlib import pyplot as plt
from matplotlib.figure import Figure


def standard_preprocess(adata: AnnData) -> AnnData:
    adata = adata[adata[:, 'volume_um3'].X > 100, :]
    adata = adata[adata[:, 'volume_um3'].X < 250, :]
    scanpy.preprocessing.log1p(adata)
    scanpy.preprocessing.scale(adata)
    return adata


def style_variable_name(raw_name: str) -> str:
    return (raw_name
            .replace("_um_local", " (local)")
            .replace("_um2_local", " (local)")
            .replace("_um3_local", " (local)")
            .replace("_local", " (local)")
            .replace("_um3", " (μm$^3$)")
            .replace("_um2", " (μm$^2$)")
            .replace("_um", " (μm)")
            .replace("_", " "))


def new_figure(size: Tuple[float, float] = (4, 3)) -> Figure:
    plt.rcParams["savefig.dpi"] = 300
    plt.rcParams["axes.edgecolor"] = "#2d3436"
    plt.rcParams["axes.linewidth"] = 1.5
    plt.rcParams["figure.autolayout"] = True
    plt.rcParams["font.family"] = "Arial"
    plt.rcParams["font.size"] = 8
    plt.rcParams['svg.fonttype'] = 'none'
    plt.rcParams["xtick.direction"] = "in"
    plt.rcParams["xtick.major.width"] = 0.75
    plt.rcParams["xtick.major.size"] = 4
    plt.rcParams["xtick.color"] = "#2d3436"
    plt.rcParams["ytick.direction"] = "in"
    plt.rcParams["ytick.major.width"] = 0.75
    plt.rcParams["ytick.major.size"] = 4
    plt.rcParams["ytick.color"] = "#2d3436"

    return plt.figure(figsize=size)
