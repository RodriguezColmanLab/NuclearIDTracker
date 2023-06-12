from typing import Tuple, Dict, Any, Union, Iterable

import numpy
import scanpy.preprocessing
from anndata import AnnData
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from numpy import ndarray

from organoid_tracker.core.experiment import Experiment

CELL_TYPE_PALETTE = {
    "ENTEROCYTE": "#100069",
    "ABSORPTIVE_PROGENY": "#0984e3",
    "SECRETORY": "#74b9ff",
    "ENTEROENDOCRINE": "#74b9ff",
    "GOBLET": "#74b9ff",
    "SECRETIVE_PROGENY": "#74b9ff",
    "WGA_PLUS": "#74b9ff",
    "PANETH": "#B60101",
    "STEM": "#26CC3C",
    "STEM_PUTATIVE": "#26CC3C",
    "UNLABELED": "#eeeeee",
    "TA": "#eeeeee",
    "NONE": "#eeeeee"
}


def get_min_max_chance_per_cell_type(experiment: Union[Experiment, Iterable[Experiment]]) -> Tuple[ndarray, ndarray]:
    probabilities = numpy.array([probabilities for position, probabilities in
                                 experiment.position_data.find_all_positions_with_data("ct_probabilities")])
    min_intensity = numpy.min(probabilities, axis=0)
    max_intensity = numpy.quantile(probabilities, q=0.95, axis=0)

    return min_intensity, max_intensity


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
