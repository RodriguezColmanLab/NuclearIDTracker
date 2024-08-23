from collections import defaultdict

import math
import matplotlib.colors
import numpy
import scanpy.plotting
import scanpy.preprocessing
import scanpy.tools
from anndata import AnnData
from enum import Enum, auto
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from numpy import ndarray
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from typing import Dict, NamedTuple

import lib_data
import lib_figures
from organoid_tracker.core.experiment import Experiment
from organoid_tracker.imaging import list_io

LDA_FILE = "../../Data/all_data.h5ad"

_DATA_FILE = "../../Data/Testing data - output - treatments.autlist"

_WINDOW_HALF_WIDTH_TIME_POINTS = 20
_BINS = numpy.arange(-10, 10, 0.4)
_COLORMAP = matplotlib.colors.LinearSegmentedColormap.from_list("histo", [(0, 0, 0, 0), (0, 0, 0, 0.7)])


class _Condition(Enum):
    CONTROL = auto()
    DAPT_CHIR = auto()
    NO_RSPONDIN = auto()
    CHIR_VPA = auto()

    @property
    def display_name(self):
        return self.name.lower().replace("_", " ")


def _get_condition(experiment: Experiment) -> _Condition:
    name = experiment.name.get_name()
    if "control" in name:
        return _Condition.CONTROL
    if "dapt chir" in name:
        return _Condition.DAPT_CHIR
    if "chir vpa" in name:
        return _Condition.CHIR_VPA
    if "EN" in name:
        return _Condition.NO_RSPONDIN
    raise ValueError("Unknown condition: " + name)


class _Dots(NamedTuple):
    x_values: ndarray
    y_values: ndarray

    def merge(self, other: "_Dots") -> "_Dots":
        return _Dots(
            x_values=numpy.concatenate([self.x_values, other.x_values]),
            y_values=numpy.concatenate([self.y_values, other.y_values])
        )

    @staticmethod
    def empty():
        return _Dots(x_values=numpy.array([], dtype=numpy.float32), y_values=numpy.array([], dtype=numpy.float32))


def _desaturate(colors: Dict[str, str]) -> Dict[str, str]:
    def desaturate_color(color: str):
        r, g, b = matplotlib.colors.to_rgb(color)

        # Desaturate
        factor = 0.4
        luma = 0.3 * r + 0.6 * g + 0.1 * b
        new_r = r + factor * (luma - r)
        new_g = g + factor * (luma - g)
        new_b = b + factor * (luma - b)
        r, g, b = new_r, new_g, new_b

        # Make brighter
        adder = 0.5 if numpy.mean([r, g, b]) < 0.5 else 0.2
        r = min(1, r + adder)
        g = min(1, g + adder)
        b = min(1, b + adder)

        return matplotlib.colors.to_hex((r, g, b))

    return dict([
        (key, desaturate_color(color)) for key, color in colors.items()
    ])


def _extract_dots(experiment: Experiment, adata: AnnData, lda: LinearDiscriminantAnalysis) -> _Dots:
    input_names = list(adata.var_names)

    all_position_data = list()
    position_names = list()
    for position in experiment.positions:
        data_array = lib_data.get_data_array(experiment.position_data, position, input_names)
        if data_array is not None and not numpy.any(numpy.isnan(data_array)):
            all_position_data.append(data_array)
            position_names.append(str(position))

    adata_treated = AnnData(numpy.array(all_position_data))
    adata_treated.var_names = input_names
    adata_treated.obs_names = position_names

    # Preprocess, but scale using the same method as used for adata
    adata_treated = lib_figures.standard_preprocess(adata_treated, filter=False, scale=False)
    adata_treated.X -= numpy.array(adata.var["mean"])
    adata_treated.X /= numpy.array(adata.var["std"])

    plot_coords = lda.transform(adata_treated.X)
    return _Dots(plot_coords[:, 0], plot_coords[:, 1])


def main():
    # Loading and preprocessing
    adata = scanpy.read_h5ad(LDA_FILE)
    adata = lib_figures.standard_preprocess(adata)

    # Do the LDA
    lda = LinearDiscriminantAnalysis()
    lda.fit(adata.X, adata.obs["cell_type_training"])

    # Load the trajectories
    experiments = list_io.load_experiment_list_file(_DATA_FILE)
    dots = defaultdict(_Dots.empty)
    for experiment in experiments:
        condition = _get_condition(experiment)
        print(experiment.name, condition)
        dots[condition] = dots[condition].merge(_extract_dots(experiment, adata, lda))

    # Plot the LDA
    figure = lib_figures.new_figure(size=(4.5, 3.5))
    conditions = list(_Condition)
    axes = numpy.array(figure.subplots(nrows=math.ceil(len(conditions) / 2), ncols=2, sharex=True, sharey=True)).flatten()
    for i in range(len(conditions)):
        is_last = i == len(conditions) - 1
        ax = axes[i]
        condition = conditions[i]
        _plot_lda(ax, lda, adata, legend=is_last)
        hist, _, _ = numpy.histogram2d(dots[condition].x_values, dots[condition].y_values, bins=(_BINS, _BINS))
        ax.hist2d(dots[condition].x_values, dots[condition].y_values,
                  bins=(_BINS, _BINS), cmap=_COLORMAP, vmax=hist.max() / 1.5)
        ax.set_aspect(1)
        ax.set_title(condition.display_name)
        ax.set_xticks([])
        ax.set_yticks([])
    plt.show()


def _plot_lda(ax: Axes, lda: LinearDiscriminantAnalysis, adata: AnnData, *, legend: bool):
    plot_coords = lda.transform(adata.X)

    # Plot the LDA
    used_cell_types = adata.obs["cell_type_training"].array.categories
    for cell_type in used_cell_types:
        depth = -1 if cell_type == "NONE" else 0
        mask = adata.obs["cell_type_training"] == cell_type
        ax.scatter(plot_coords[mask, 0], plot_coords[mask, 1],
                   s=20, lw=0, zorder=depth,
                   color=lib_figures.CELL_TYPE_PALETTE[cell_type], label=lib_figures.style_cell_type_name(cell_type))
    if legend:
        ax.legend(
            loc='center left', bbox_to_anchor=(1, 0.5)
        )


if __name__ == "__main__":
    main()
