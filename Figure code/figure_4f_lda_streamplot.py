from typing import NamedTuple, Iterable

import numpy
import numpy as np
import pandas
import scanpy.plotting
import scanpy.preprocessing
import scanpy.tools
from anndata import AnnData
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from numpy import ndarray
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

import lib_data
import lib_figures
import lib_streamplot
import lib_models
from organoid_tracker.core.experiment import Experiment
from organoid_tracker.core.position import Position
from organoid_tracker.imaging import list_io

LDA_FILE = "../../Data/all_data.h5ad"
_MODEL_FOLDER = r"../../Data/Models/epochs-1-neurons-0"

_DATA_FILE_CONTROL = "../../Data/Tracking data as controls/Dataset.autlist"
_DATA_FILE_REGENERATION = "../../Data/Stem cell regeneration/Dataset - post DT removal.autlist"
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
            self._dy_sums[y_coord, x_coord] += -dy / dt
            self._counts[y_coord, x_coord] += 1

    def plot(self, ax: Axes):
        # Calculate speeds
        dx_values = self._dx_sums / numpy.clip(self._counts, 1, None)
        dy_values = self._dy_sums / numpy.clip(self._counts, 1, None)

        # Don't show arrows where we have less than 2 trajectories
        dx_values[self._counts < 5] = 0
        dy_values[self._counts < 5] = 0

        # speed = np.sqrt(dx_values ** 2 + dy_values ** 2)
        #
        # # Cap speed by a maximum value
        # max_speed = 0.1
        # factor = numpy.where(speed > max_speed, max_speed / (speed + 0.00001), numpy.ones_like(speed))
        # dx_values *= factor
        # dy_values *= factor

        lib_streamplot.streamplot(ax, self._x_coords, self._y_coords, dx_values, dy_values, density=1.5, color="white",
                                  linewidth=4, maxlength=0.2, integration_direction="forward")
        lib_streamplot.streamplot(ax, self._x_coords, self._y_coords, dx_values, dy_values, density=1.5, color="black",
                                  linewidth=1, maxlength=0.2, integration_direction="forward")


def _extract_trajectories(experiment: Experiment, adata: AnnData, lda: LinearDiscriminantAnalysis, streamplot: _Streamplot):
    input_names = list(adata.var_names)
    resolution = experiment.images.resolution()

    for track in experiment.links.find_all_tracks():
        all_position_data = list()
        position_names = list()
        time_h = list()
        positions = list(track.positions())
        parent_tracks = track.get_previous_tracks()
        if len(parent_tracks) == 1:
            positions = list(parent_tracks.pop().positions()) + positions
        for position in positions:
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

    # Set up the figures
    figure = lib_figures.new_figure(size=(8.5, 4.5))
    ax_control, ax_regeneration = figure.subplots(nrows=1, ncols=2, sharex=True, sharey=True)
    ax_control.set_ylim(-5, 5)
    ax_control.set_xlim(-5, 5)
    ax_control.set_title("De-novo growth")
    ax_regeneration.set_title("Regeneration")

    plot_coords_control = _plot_experiments(adata, ax_control, list(list_io.load_experiment_list_file(_DATA_FILE_CONTROL, load_images=False)))
    plot_coords_regen = _plot_experiments(adata, ax_regeneration, list(list_io.load_experiment_list_file(_DATA_FILE_REGENERATION, load_images=False)))

    # Add coords to the other plot too, for reference, as grayed out points
    ax_control.scatter(plot_coords_regen[:, 0], plot_coords_regen[:, 1], s=3, lw=0, c="#e5e5e5", zorder=-1)
    ax_regeneration.scatter(plot_coords_control[:, 0], plot_coords_control[:, 1], s=3, lw=0, c="#e5e5e5", zorder=-1)

    ax_control.set_aspect(1)
    ax_regeneration.set_aspect(1)
    plt.show()


def _plot_experiments(adata_training: AnnData, ax: Axes, experiments: Iterable[Experiment]) -> ndarray:
    """Plots the LDA streamplot for the given experiments. Returns the coordinates of the points."""
    # Do the LDA
    lda = LinearDiscriminantAnalysis()
    lda.fit(adata_training.X, adata_training.obs["cell_type_training"])

    # Load the trajectories
    streamplot = _Streamplot()
    for experiment in experiments:
        _extract_trajectories(experiment, adata_training, lda, streamplot)
        break

    # Predict cell types of all cells
    adata_predictions = _get_adata_predictions_selected_time_points(experiments)
    model = lib_models.load_model(_MODEL_FOLDER)
    parameters_name = model.get_input_output().input_mapping
    parameters_order = numpy.array([parameters_name.index(name) for name in adata_predictions.var_names])
    result = model.predict(adata_predictions.X[:, parameters_order])
    cell_types = model.get_input_output().cell_type_mapping
    colors = [lib_figures.get_mixed_cell_type_color(cell_types, result[i]) for i in range(len(result))]

    # Continue processing for LDA plot
    adata_predictions = lib_figures.standard_preprocess(adata_predictions, filter=False, scale=False)
    adata_predictions.X -= numpy.array(adata_training.var["mean"])
    adata_predictions.X /= numpy.array(adata_training.var["std"])
    plot_coords = lda.transform(adata_predictions.X)
    ax.scatter(plot_coords[:, 0], plot_coords[:, 1], s=12, lw=0, c=colors)
    streamplot.plot(ax)

    return plot_coords


def _get_adata_predictions_selected_time_points(experiments: Iterable[Experiment]) -> AnnData:
    """Gets the raw features, so that we can create a heatmap. Cells are indexed by _get_cell_key."""
    data_array = list()
    cell_type_list = list()
    organoid_list = list()
    cell_names = list()
    # Collect position data for last 10 time points of each experiment
    for experiment in experiments:
        print("Loading", experiment.name)

        position_data = experiment.position_data

        for time_point in experiment.positions.time_points():
            if time_point.time_point_number() < experiment.positions.last_time_point_number() - 5:
                continue  # Prevent accumulating too many points
            for position in experiment.positions.of_time_point(time_point):
                position_data_array = lib_data.get_data_array(position_data, position, lib_data.STANDARD_METADATA_NAMES)
                cell_type = position_data.get_position_data(position, "type")
                if position_data_array is not None and cell_type is not None:
                    data_array.append(position_data_array)
                    cell_type_list.append(cell_type)
                    organoid_list.append(experiment.name.get_name())
                    cell_names.append(_get_cell_key(experiment, position))
    data_array = numpy.array(data_array, dtype=numpy.float32)

    adata = AnnData(data_array)
    adata.var_names = lib_data.STANDARD_METADATA_NAMES
    adata.obs_names = cell_names
    adata.obs["cell_type"] = pandas.Categorical(cell_type_list)
    adata.obs["organoid"] = pandas.Categorical(organoid_list)

    return adata


def _get_cell_key(experiment: Experiment, position: Position) -> str:
    return f"{experiment.name}-{int(position.x)}-{int(position.y)}-{int(position.z)}"


if __name__ == "__main__":
    main()
