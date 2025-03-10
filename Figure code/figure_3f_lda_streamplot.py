import math
import numpy
import numpy as np
import scanpy.plotting
import scanpy.preprocessing
import scanpy.tools
from anndata import AnnData
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from numpy import ndarray
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from typing import NamedTuple

import lib_data
import lib_figures
import lib_streamplot
from organoid_tracker.core.experiment import Experiment
from organoid_tracker.imaging import list_io

LDA_FILE = "../../Data/all_data.h5ad"

_DATA_FILE = "../../Data/Tracking data as controls/Dataset.autlist"
# x20190926pos01 (first one we tried), x20190817pos01, x20200614pos10

class _Trajectory(NamedTuple):
    x_values: ndarray
    y_values: ndarray
    time_h: ndarray

    def resample_5h(self) -> "_Trajectory":
        indices = (self.time_h / 5).astype(numpy.int32)

        x_values_new = list()
        y_values_new = list()
        time_h_new = list()
        for i in range(indices.max() + 1):
            if len(self.time_h[indices == i]) == 0:
                continue  # No values for that time period
            x_values_new.append(self.x_values[indices == i].mean())
            y_values_new.append(self.y_values[indices == i].mean())
            time_h_new.append(int(self.time_h[indices == i][0] / 5) * 5)

        return _Trajectory(
            x_values=numpy.array(x_values_new),
            y_values=numpy.array(y_values_new),
            time_h=numpy.array(time_h_new))


class _Streamplot:

    _x_coords: ndarray
    _y_coords: ndarray
    _dx_sums: ndarray
    _dy_sums: ndarray
    _counts: ndarray

    _half_width: float
    _count: int

    def __init__(self):
        self._half_width = 5
        self._count = 25
        self._x_coords, self._y_coords = numpy.meshgrid(
            numpy.linspace(-self._half_width, self._half_width, self._count),
            numpy.linspace(-self._half_width, self._half_width, self._count))
        self._dx_sums = numpy.zeros_like(self._x_coords)
        self._dy_sums = numpy.zeros_like(self._x_coords)
        self._counts = numpy.zeros_like(self._x_coords)

    def add_trajectory(self, trajectory: _Trajectory):
        for i in range(1, len(trajectory.x_values)):
            x_start, y_start = trajectory.x_values[i - 1], trajectory.y_values[i - 1]
            x_offset = x_start + self._half_width  # So -3 becomes 0
            y_offset = y_start + self._half_width
            x_coord = int(x_offset / (2 * self._half_width) * self._count)
            y_coord = int(y_offset / (2 * self._half_width) * self._count)
            if x_coord < 0 or x_coord >= self._count or y_coord < 0 or y_coord >= self._count:
                continue

            dx = trajectory.x_values[i] - x_start
            dy = trajectory.y_values[i] - y_start
            dt = trajectory.time_h[i] - trajectory.time_h[i - 1]
            self._dx_sums[y_coord, x_coord] += dx / dt
            self._dy_sums[y_coord, x_coord] += dy / dt
            self._counts[y_coord, x_coord] += 1

    def plot(self, ax: Axes):
        # Calculate speeds
        dx_values = self._dx_sums / numpy.clip(self._counts, 1, None)
        dy_values = self._dy_sums / numpy.clip(self._counts, 1, None)

        speed = np.sqrt(dx_values ** 2 + dy_values ** 2)

        # Cap speed by a maximum value
        max_speed = 0.1
        factor = numpy.where(speed > max_speed, max_speed / (speed + 0.00001), numpy.ones_like(speed))
        dx_values *= factor
        dy_values *= factor

        lib_streamplot.streamplot(ax, self._x_coords, self._y_coords, dx_values, dy_values, density=1.5, color="white",
                                  linewidth=2, maxlength=0.12, integration_direction="forward")
        lib_streamplot.streamplot(ax, self._x_coords, self._y_coords, dx_values, dy_values, density=1.5, color="black",
                                  linewidth=1, maxlength=0.12, integration_direction="forward")


def _extract_trajectories(experiment: Experiment, adata: AnnData, lda: LinearDiscriminantAnalysis, streamplot: _Streamplot):
    input_names = list(adata.var_names)
    resolution = experiment.images.resolution()

    for track in experiment.links.find_ending_tracks():
        all_position_data = list()
        position_names = list()
        time_h = list()
        for position in track.positions(connect_to_previous_track=True):
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
        trajectory = _Trajectory(plot_coords[:, 0], plot_coords[:, 1],
                                 numpy.array(adata_track.obs["time_h"])).resample_5h()
        streamplot.add_trajectory(trajectory)


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

    streamplot = _Streamplot()
    for experiment in experiments:
        _extract_trajectories(experiment, adata, lda, streamplot)
        break

    # Plot the LDA
    figure = lib_figures.new_figure(size=(3.5, 2.5))
    ax: Axes = figure.gca()
    ax.set_ylim(-4, 7)
    ax.set_xlim(-5, 5)
    _plot_lda(ax, lda, adata)

    streamplot.plot(ax)



    ax.set_aspect(1)
    plt.show()


def _plot_lda(ax: Axes, lda: LinearDiscriminantAnalysis, adata: AnnData):
    plot_coords = lda.transform(adata.X)

    # Plot the LDA
    used_cell_types = adata.obs["cell_type_training"].array.categories
    colors = [lib_figures.CELL_TYPE_PALETTE[cell_type] for cell_type in adata.obs["cell_type_training"]]
    ax.scatter(plot_coords[:, 0], plot_coords[:, 1], s=15, lw=0, c=colors)

    # Add legend for all the cell types
    for cell_type in used_cell_types:
        ax.scatter([], [], label=lib_figures.style_cell_type_name(cell_type), s=15, lw=0,
                   color=lib_figures.CELL_TYPE_PALETTE[cell_type])
    ax.set_title("Linear Discriminant Analysis")
    ax.legend()


if __name__ == "__main__":
    main()
