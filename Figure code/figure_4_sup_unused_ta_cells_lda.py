from collections import defaultdict

import math
import matplotlib.colors
import numpy
import pandas
import scanpy.plotting
import scanpy.preprocessing
import scanpy.tools
from anndata import AnnData
from enum import Enum, auto
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from numpy import ndarray
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from typing import Dict, NamedTuple, Iterable

import lib_data
import lib_figures
from organoid_tracker.core.experiment import Experiment
from organoid_tracker.core.position import Position
from organoid_tracker.imaging import list_io
from organoid_tracker.position_analysis import position_markers

LDA_FILE = "../../Data/all_data.h5ad"

_DATA_FILE_REGENERATION = "../../Data/Stem cell regeneration/Dataset - post DT removal.autlist"
_DATA_FILE_CONTROL = "../../Data/Tracking data as controls/Dataset.autlist"

_X_LIMITS = (-5, 7)
_Y_LIMITS = (4, -9)

_TA_MIN_SCORE = 0.4
_TA_MAX_SCORE = 0.55

_TIME_WINDOW_H = 8


def _desaturate(colors: Dict[str, str]) -> Dict[str, str]:
    def desaturate_color(color: str):
        r, g, b = matplotlib.colors.to_rgb(color)

        # Desaturate
        factor = 0.8
        luma = 0.3 * r + 0.6 * g + 0.1 * b
        new_r = r + factor * (luma - r)
        new_g = g + factor * (luma - g)
        new_b = b + factor * (luma - b)
        r, g, b = new_r, new_g, new_b

        # Make brighter
        adder = 0.3 if numpy.mean([r, g, b]) < 0.5 else 0.1
        r = min(1, r + adder)
        g = min(1, g + adder)
        b = min(1, b + adder)

        return matplotlib.colors.to_hex((r, g, b))

    return dict([
        (key, desaturate_color(color)) for key, color in colors.items()
    ])


def _get_adata_predictions(experiments: Iterable[Experiment]) -> AnnData:
    """Gets the raw features, so that we can create a heatmap. Cells are indexed by _get_cell_key."""
    data_array = list()
    cell_type_list = list()
    organoid_list = list()
    stem_to_ec_score_list = list()
    time_h_list = list()
    cell_names = list()
    # Collect position data for last 10 time points of each experiment
    for experiment in experiments:
        print("Loading", experiment.name)

        cell_types = experiment.global_data.get_data("ct_probabilities")

        position_data = experiment.positions
        timings = experiment.images.timings()

        for position in experiment.positions:
            position_data_array = lib_data.get_data_array(position_data, position, lib_data.STANDARD_METADATA_NAMES)
            cell_type = position_data.get_position_data(position, "type")
            probabilities = position_data.get_position_data(position, "ct_probabilities")
            stem_to_enterocyte_score = lib_data.find_stem_to_ec_location(cell_types, probabilities)
            time_h = timings.get_time_h_since_start(position.time_point_number())

            if position_data_array is not None and cell_type is not None and stem_to_enterocyte_score is not None:
                data_array.append(position_data_array)
                cell_type_list.append(cell_type)
                organoid_list.append(experiment.name.get_name())
                stem_to_ec_score_list.append(stem_to_enterocyte_score)
                cell_names.append(_get_cell_key(experiment, position))
                time_h_list.append(time_h)
    data_array = numpy.array(data_array, dtype=numpy.float32)

    adata = AnnData(data_array)
    adata.var_names = lib_data.STANDARD_METADATA_NAMES
    adata.obs_names = cell_names
    adata.obs["cell_type"] = pandas.Categorical(cell_type_list)
    adata.obs["organoid"] = pandas.Categorical(organoid_list)
    adata.obs["stem_to_ec_score"] = stem_to_ec_score_list
    adata.obs["time_h"] = time_h_list

    return adata


def _get_cell_key(experiment: Experiment, position: Position) -> str:
    return f"{experiment.name}-{int(position.x)}-{int(position.y)}-{int(position.z)}"


def _filter_for_time(adata: AnnData, min_time_h: float, max_time_h: float) -> AnnData:
    """Filters the AnnData object for cells within the given time window. The time is given in hours.
    The minimum is inclusive, the maximum is exclusive."""
    return adata[
        (adata.obs["time_h"] >= min_time_h) &
        (adata.obs["time_h"] < max_time_h)
        ]


def main():
    # Loading and preprocessing
    adata = scanpy.read_h5ad(LDA_FILE)
    adata = lib_figures.standard_preprocess(adata, log1p=False, scale=False)

    # Remove cells that we cannot train on
    adata = adata[adata.obs["cell_type_training"] != "NONE"]

    # Do the LDA
    lda = LinearDiscriminantAnalysis()
    lda.fit(adata.X, adata.obs["cell_type_training"])

    # Load the features and immunostaining results
    experiments_regeneration = list_io.load_experiment_list_file(_DATA_FILE_REGENERATION, load_images=False)
    experiments_control = list_io.load_experiment_list_file(_DATA_FILE_CONTROL, load_images=False)

    adata_regeneration = _get_adata_predictions(experiments_regeneration)
    adata_regeneration = lib_figures.standard_preprocess(adata_regeneration, log1p=False, scale=False)
    adata_regeneration = _filter_for_ta(adata_regeneration)

    adata_control = _get_adata_predictions(experiments_control)
    adata_control = lib_figures.standard_preprocess(adata_control, log1p=False, scale=False)
    adata_control = _filter_for_ta(adata_control)

    # Plot the LDA
    figure = lib_figures.new_figure(size=(8, 7), dpi=600)
    axes = numpy.array(figure.subplots(nrows=3, ncols=3, sharex=True, sharey=True))
    for time_id in range(axes.shape[0]):
        for column_id in range(axes.shape[1]):
            colors = _desaturate(lib_figures.CELL_TYPE_PALETTE)
            ax = axes[time_id, column_id]


            # Special case for first column: use original colors or don't draw at all
            if column_id == 0:
                if time_id == 0:
                    colors = lib_figures.CELL_TYPE_PALETTE
                else:
                    # Keep plot empty
                    ax.axis("off")
                    continue

            _plot_lda(ax, lda, adata, colors=colors)

            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_ylim(_Y_LIMITS)
            ax.set_xlim(_X_LIMITS)
            ax.set_aspect("equal")

        ax_control = axes[time_id, 1]
        ax_regeneration = axes[time_id, 2]

        min_time_h = time_id * _TIME_WINDOW_H
        max_time_h = (time_id + 1) * _TIME_WINDOW_H

        ax_control.set_title(f"TA cells - Control - {min_time_h}-{max_time_h} h")
        _put_on_lda(ax_control, lda, _filter_for_time(adata_control, min_time_h, max_time_h))

        ax_regeneration.set_title(f"TA cells - Regeneration - {min_time_h}-{max_time_h} h")
        _put_on_lda(ax_regeneration, lda, _filter_for_time(adata_regeneration, min_time_h, max_time_h))

    figure.tight_layout()
    plt.show()


def _put_on_lda(ax: Axes, lda: LinearDiscriminantAnalysis, adata: AnnData):
    plot_coords = lda.transform(adata.X)
    ax.scatter(plot_coords[:, 0], plot_coords[:, 1], s=8, lw=0, c='black', alpha=0.01)


def _filter_for_ta(adata: AnnData) -> AnnData:
    return adata[
        (adata.obs["stem_to_ec_score"] >= _TA_MIN_SCORE) &
        (adata.obs["stem_to_ec_score"] <= _TA_MAX_SCORE)
        ]


def _plot_lda(ax: Axes, lda: LinearDiscriminantAnalysis, adata: AnnData, *,
              colors: Dict[str, str] = lib_figures.CELL_TYPE_PALETTE, s: float = 8):
    plot_coords = lda.transform(adata.X)

    colors = [colors[cell_type] for cell_type in adata.obs["cell_type_training"]]
    ax.scatter(plot_coords[:, 0], plot_coords[:, 1], s=s, lw=0, c=colors)


if __name__ == "__main__":
    main()
