import math
from typing import Dict, List, NamedTuple, Iterable

import matplotlib.colors
import numpy
import pandas
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
import lib_models
from organoid_tracker.core import TimePoint
from organoid_tracker.core.experiment import Experiment
from organoid_tracker.core.position import Position
from organoid_tracker.imaging import list_io

LDA_FILE = "../../Data/all_data.h5ad"
_MODEL_FOLDER = r"../../Data/Models/epochs-1-neurons-0"

_TRAJECTORIES_DATA_FILE = "../../Data/Predicted data.autlist"
_TRAJECTORIES_EXPERIMENT_NAME = "x20190926pos01"

_BACKGROUND_FILE = "../../Data/Tracking data as controls/Dataset.autlist"

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


def main():
    # Loading and preprocessing
    adata_training = scanpy.read_h5ad(LDA_FILE)
    adata_training = lib_figures.standard_preprocess(adata_training)

    # Remove cells that we cannot train on
    adata_training = adata_training[adata_training.obs["cell_type_training"] != "NONE"]

    # Do the LDA
    lda = LinearDiscriminantAnalysis()
    lda.fit(adata_training.X, adata_training.obs["cell_type_training"])

    # Extract trajectories
    trajectories = list()
    for experiment in list_io.load_experiment_list_file(_TRAJECTORIES_DATA_FILE, load_images=False):
        if experiment.name.get_name() == _TRAJECTORIES_EXPERIMENT_NAME:
            _extract_trajectories(experiment, adata_training, lda, trajectories)
            break

    # Extract plot background
    adata_predictions = _get_adata_predictions_selected_time_points(list_io.load_experiment_list_file(_BACKGROUND_FILE, load_images=False))
    model = lib_models.load_model(_MODEL_FOLDER)
    parameters_name = model.get_input_output().input_mapping
    parameters_order = numpy.array([parameters_name.index(name) for name in adata_predictions.var_names])
    result = model.predict(adata_predictions.X[:, parameters_order])
    cell_types = model.get_input_output().cell_type_mapping
    colors = [lib_figures.get_mixed_cell_type_color(cell_types, result[i]) for i in range(len(result))]

    # Plot the LDA
    figure = lib_figures.new_figure(size=(3.5, 2.5))
    ax: Axes = figure.gca()

    # Continue processing for LDA background
    adata_predictions = lib_figures.standard_preprocess(adata_predictions, filter=False, scale=False)
    adata_predictions.X -= numpy.array(adata_training.var["mean"])
    adata_predictions.X /= numpy.array(adata_training.var["std"])
    plot_coords = lda.transform(adata_predictions.X)
    ax.scatter(plot_coords[:, 0], plot_coords[:, 1], s=12, lw=0, c=colors)

    # Plot trajectories on top of that
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
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    plt.show()



if __name__ == "__main__":
    main()
