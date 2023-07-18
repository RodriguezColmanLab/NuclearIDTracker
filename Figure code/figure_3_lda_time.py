import math
from typing import Dict, List, NamedTuple

import matplotlib.colors
import numpy
import pandas
import scanpy.plotting
import scanpy.preprocessing
import scanpy.tools
import scipy
from anndata import AnnData
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from numpy import ndarray
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

import lib_data
import lib_figures
from organoid_tracker.core.experiment import Experiment
from organoid_tracker.core.position import Position
from organoid_tracker.imaging import list_io
from organoid_tracker.util import mpl_helper

LDA_FILE = "../../Data/all_data.h5ad"

_DATA_FILE = "../../Data/Predicted data.autlist"
_EXPERIMENT_NAME = "x20190926pos01"
_PLOTTED_LINEAGE_TREES = [Position(192.02, 299.39, 5, time_point_number=1)]
_WINDOW_HALF_WIDTH_TIME_POINTS = 20
_TIME_COLORMAP = matplotlib.colors.LinearSegmentedColormap.from_list("dark_gray", ["#000000", "#666666"])

class _Line(NamedTuple):
    x_values: ndarray
    y_values: ndarray
    time_h: ndarray
    names: List[str]

    def resample_5h(self) -> "_Line":
        indices = (self.time_h / 7.5).astype(numpy.int32)

        x_values_new = list()
        y_values_new = list()
        time_h_new = list()
        names_new = list()
        for i in range(indices.max() + 1):
            x_values_new.append(self.x_values[indices == i].mean())
            y_values_new.append(self.y_values[indices == i].mean())
            time_h_new.append(self.time_h[indices == i].min())
            names_new.append("")

        # # Spline interpolation
        # k = 3 if len(x_values_new) > 3 else 1
        # spline, _ = scipy.interpolate.splprep([x_values_new, y_values_new], k=k)
        # points = scipy.interpolate.splev(numpy.arange(0, 1.01, 0.05), spline)
        # x_values_new = points[0]
        # y_values_new = points[1]
        # time_h_new = [1] * len(y_values_new)

        return _Line(
            x_values=numpy.array(x_values_new),
            y_values=numpy.array(y_values_new),
            time_h=numpy.array(time_h_new),
            names=self.names)


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
        adder = 0.5
        r = min(1, r + adder)
        g = min(1, g + adder)
        b = min(1, b + adder)

        return matplotlib.colors.to_hex((r, g, b))

    return dict([
        (key, desaturate_color(color)) for key, color in colors.items()
    ])


def _extract_trajectories(experiment: Experiment, adata: AnnData, lda: LinearDiscriminantAnalysis,
                          trajectories: List[_Line]):
    input_names = list(adata.var_names)
    resolution = experiment.images.resolution()

    for starting_track in experiment.links.find_starting_tracks():
        if starting_track.find_first_position() not in _PLOTTED_LINEAGE_TREES:
            continue

        for track in starting_track.find_all_descending_tracks(include_self=True):
            if len(track.get_next_tracks()) > 0:
                continue  # Only iterate over ending tracks

            all_position_data = list()
            position_names = list()
            time_h = list()
            for position in experiment.links.iterate_to_past(track.find_last_position()):
                data_array = lib_data.get_data_array(experiment.position_data, position, input_names)
                if data_array is not None and not numpy.any(numpy.isnan(data_array)):
                    all_position_data.append(data_array)
                    position_names.append(str(position))
                    time_h.append(position.time_point_number() * resolution.time_point_interval_h)

            if len(all_position_data) < 2:
                continue  # Not enough data for adata object
            adata_track = AnnData(numpy.array(all_position_data))
            adata_track.var_names = input_names
            adata_track.obs_names = position_names
            adata_track.obs["time_h"] = time_h

            # Preprocess, but scale using the same method as used for adata
            adata_track = lib_figures.standard_preprocess(adata_track, filter=False, scale=False)
            adata_track.X -= numpy.array(adata.var["mean"])
            adata_track.X /= numpy.array(adata.var["std"])

            plot_coords = lda.transform(adata_track.X)
            trajectories.append(_Line(plot_coords[:, 0], plot_coords[:, 1],
                                      numpy.array(adata_track.obs["time_h"]),
                                      list(adata_track.obs_names))
                                .resample_5h())


def main():
    # Loading and preprocessing
    adata = scanpy.read_h5ad(LDA_FILE)
    adata = lib_figures.standard_preprocess(adata, filter=False)

    # Remove cells that we cannot train on
    adata = adata[adata.obs["cell_type_training"] != "NONE"]

    # Do the LDA
    lda = LinearDiscriminantAnalysis()
    lda.fit(adata.X, adata.obs["cell_type_training"])

    # Load the trajectories
    experiments = list_io.load_experiment_list_file(_DATA_FILE)

    trajectories = list()
    for experiment in experiments:
        if experiment.name.get_name() == _EXPERIMENT_NAME:
            _extract_trajectories(experiment, adata, lda, trajectories)
            break

    # Plot the LDA
    figure = lib_figures.new_figure(size=(3.5, 2.5))
    ax: Axes = figure.gca()
    _plot_lda(ax, lda, adata)
    for line in trajectories:
        colors = _TIME_COLORMAP(line.time_h / line.time_h.max())
        mpl_helper.plot_multicolor(ax, line.x_values, line.y_values, colors=colors, linewidth=1)

    plt.show()


def _plot_lda(ax: Axes, lda: LinearDiscriminantAnalysis, adata: AnnData):
    plot_coords = lda.transform(adata.X)
    background_palette = _desaturate(lib_figures.CELL_TYPE_PALETTE)

    # Plot the LDA
    ax.scatter(plot_coords[:, 0], plot_coords[:, 1],
               alpha=0.8, s=15, lw=0,
               color=[background_palette[adata.obs["cell_type_training"][i]] for i in
                      range(len(adata.obs["cell_type_training"]))])
    used_cell_types = adata.obs["cell_type_training"].array.categories
    ax.set_title("Linear Discriminant Analysis")
    ax.legend(handles=[
        Line2D([0], [0], marker='o', alpha=0.8,
               color=lib_figures.CELL_TYPE_PALETTE[cell_type],
               label=lib_figures.style_cell_type_name(cell_type),
               markersize=math.sqrt(15), lw=0)
        for cell_type in used_cell_types],
        loc='center left', bbox_to_anchor=(1, 0.5)
    )


if __name__ == "__main__":
    main()
