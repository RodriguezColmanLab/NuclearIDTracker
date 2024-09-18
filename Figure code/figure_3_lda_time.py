import math
from typing import Dict, List, NamedTuple

import matplotlib.colors
import numpy
import scanpy.plotting
import scanpy.preprocessing
import scanpy.tools
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

LDA_FILE = "../../Data/all_data.h5ad"

_DATA_FILE = "../../Data/Predicted data.autlist"
_EXPERIMENT_NAME = "x20190926pos01"

# Must be the last position in time, as we iterate back
_PLOTTED_POSITIONS = [Position(89.73, 331.37, 5.00, time_point_number=331),
                      Position(125.56, 294.27, 12.00, time_point_number=331),
                      Position(234.63, 343.08, 8.00, time_point_number=331)]


class _Line(NamedTuple):
    x_values: ndarray
    y_values: ndarray
    time_h: ndarray
    label: int

    def resample_5h(self) -> "_Line":
        indices = (self.time_h / 5).astype(numpy.int32)

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
            label=self.label)


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


def _extract_trajectories(experiment: Experiment, adata: AnnData, lda: LinearDiscriminantAnalysis,
                          trajectories: List[_Line]):
    input_names = list(adata.var_names)
    resolution = experiment.images.resolution()

    for i, position in enumerate(_PLOTTED_POSITIONS):
        ending_track = experiment.links.get_track(position)
        if ending_track is None:
            raise ValueError(f"Position {position} has no track")

        all_position_data = list()
        position_names = list()
        time_h = list()
        for position in experiment.links.iterate_to_past(ending_track.find_last_position()):
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
        adata_track.X = numpy.clip(adata_track.X, -2, 2)

        plot_coords = lda.transform(adata_track.X)
        trajectories.append(_Line(plot_coords[:, 0], plot_coords[:, 1],
                                  numpy.array(adata_track.obs["time_h"]),
                                  i + 1)
                            .resample_5h())


def main():
    # Loading and preprocessing
    adata = scanpy.read_h5ad(LDA_FILE)
    adata = lib_figures.standard_preprocess(adata)

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
        # Plot a line
        ax.plot(line.x_values, line.y_values, color="#636e72", linewidth=1.5, zorder=4)
        # Plot dots along the line
        ax.scatter(line.x_values[:-1], line.y_values[:-1], color="#636e72", s=6, zorder=5)
        # Except for the last dot, where we plot an arrow (the coords are the line part of the arrow)
        ax.arrow(line.x_values[-2], line.y_values[-2],
                 (line.x_values[-1] - line.x_values[-2]),
                 (line.y_values[-1] - line.y_values[-2]),
                 head_width=0.5, head_length=0.7, width=0.01, linewidth=0, color="black", zorder=6)
        ax.text(line.x_values[-1], line.y_values[-1], str(line.label))

    ax.set_aspect(1)
    plt.show()


def _plot_lda(ax: Axes, lda: LinearDiscriminantAnalysis, adata: AnnData):
    plot_coords = lda.transform(adata.X)

    # Plot the LDA
    used_cell_types = adata.obs["cell_type_training"].array.categories
    for cell_type in used_cell_types:
        depth = 2 if cell_type == "NONE" else 3  # The lower, the more in the background
        mask = adata.obs["cell_type_training"] == cell_type
        ax.scatter(plot_coords[mask, 0], plot_coords[mask, 1],
                   s=10, lw=0, zorder=depth,
                   color=lib_figures.CELL_TYPE_PALETTE[cell_type], label=lib_figures.style_cell_type_name(cell_type))
    ax.set_title("Linear Discriminant Analysis")
    ax.legend()


if __name__ == "__main__":
    main()
