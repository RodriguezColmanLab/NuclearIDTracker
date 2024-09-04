from typing import Tuple, Dict, Any, Union, Iterable, List

import numpy
from anndata import AnnData
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from numpy import ndarray

from organoid_tracker.core.experiment import Experiment

CELL_TYPE_PALETTE = {
    "ENTEROCYTE": "#507B9D",
    "KRT20_POSITIVE": "#507B9D",
    "ABSORPTIVE_PROGENY": "#0984e3",
    "SECRETORY": "#507B9D",
    "ENTEROENDOCRINE": "#507B9D",
    "OTHER_SECRETORY": "#507B9D",
    "MATURE_GOBLET": "#DA5855",
    "SECRETIVE_PROGENY": "#507B9D",
    "WGA_PLUS": "#74b9ff",
    "PANETH": "#F2A18E",
    "STEM": "#B6E0A0",
    "DOUBLE_NEGATIVE": "#B6E0A0",
    "STEM_PUTATIVE": "#B7E1A1",
    "STEM_FETAL": "#77A2B5",
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


def standard_preprocess(adata: AnnData, scale: bool = True, log1p: bool = True, filter: bool = True) -> AnnData:
    import scanpy.preprocessing
    if filter:
        adata = adata[adata[:, 'volume_um3'].X > 200, :]
        adata = adata[adata[:, 'volume_um3'].X < 400, :]
    if log1p:
        scanpy.preprocessing.log1p(adata)
    if scale:
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
    plt.rcParams["axes.linewidth"] = 1
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


def style_cell_type_name(cell_type_name: str) -> str:
    return (cell_type_name.lower().replace("mature_", "").replace("_", " ")
            .replace("paneth", "Paneth").replace("stem fetal", "fetal stem"))


def get_mixed_cell_type_color(cell_type_names: List[str], cell_type_probabilities: List[float]) -> str:
    highest_probability = max(cell_type_probabilities)
    probability_max_plotting = highest_probability
    probability_min_plotting = highest_probability * 0.97

    scaled_probabilities = [
        _clip((probability - probability_min_plotting) / (probability_max_plotting - probability_min_plotting), 0, 1)
        for probability in cell_type_probabilities
    ]
    scaled_probabilities = [
        probability / sum(scaled_probabilities) for probability in scaled_probabilities
    ]

    d_items = [
        [CELL_TYPE_PALETTE[cell_type] for cell_type in cell_type_names],
        scaled_probabilities
    ]

    red = int(sum([int(k[1:3], 16) * v for k, v in zip(*d_items)]))
    green = int(sum([int(k[3:5], 16) * v for k, v in zip(*d_items)]))
    blue = int(sum([int(k[5:7], 16) * v for k, v in zip(*d_items)]))

    def zero_pad(x):
        return x if len(x) == 2 else '0' + x

    return "#" + zero_pad(hex(red)[2:]) + zero_pad(hex(green)[2:]) + zero_pad(hex(blue)[2:])


def _clip(number: float, min_number: float, max_number: float) -> float:
    """Clips a number to a range."""
    if number < min_number:
        return min_number
    if number > max_number:
        return max_number
    return number

