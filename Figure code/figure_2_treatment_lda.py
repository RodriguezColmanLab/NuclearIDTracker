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

_DATA_FILE_PREDICTIONS = "../../Data/Testing data - predictions - treatments - fixed.autlist"
_DATA_FILE_STAINING = "../../Data/Immunostaining conditions.autlist"
_STAINING_CELL_TYPES = {
    None: "double-negative",
    "PANETH": "lysozym-positive",
    "ENTEROCYTE": "KRT20-positive"
}

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
    treatment_list = list()
    cell_names = list()
    # Collect position data for last 10 time points of each experiment
    for experiment in experiments:
        print("Loading", experiment.name)
        treatment = _get_condition(experiment)

        position_data = experiment.position_data

        for position in experiment.positions:
            position_data_array = lib_data.get_data_array(position_data, position, lib_data.STANDARD_METADATA_NAMES)
            cell_type = position_data.get_position_data(position, "type")
            if position_data_array is not None and cell_type is not None:
                data_array.append(position_data_array)
                cell_type_list.append(cell_type)
                organoid_list.append(experiment.name.get_name())
                cell_names.append(_get_cell_key(experiment, position))
                treatment_list.append(treatment.display_name)
    data_array = numpy.array(data_array, dtype=numpy.float32)

    adata = AnnData(data_array)
    adata.var_names = lib_data.STANDARD_METADATA_NAMES
    adata.obs_names = cell_names
    adata.obs["cell_type"] = pandas.Categorical(cell_type_list)
    adata.obs["organoid"] = pandas.Categorical(organoid_list)
    adata.obs["treatment"] = pandas.Categorical(treatment_list)

    return adata


def _get_cell_key(experiment: Experiment, position: Position) -> str:
    return f"{experiment.name}-{int(position.x)}-{int(position.y)}-{int(position.z)}"


def _get_staining_table(experiments: Iterable[Experiment]) -> pandas.DataFrame:
    """Get a dictionary of cell types for each position in each experiment."""

    cell_types_predicted = dict()
    for experiment in experiments:
        print(f"Working on {experiment.name.get_name()}...")
        position_data = experiment.position_data

        for time_point in experiment.positions.time_points():
            for position in experiment.positions.of_time_point(time_point):
                position_type = position_markers.get_position_type(position_data, position)
                position_type = _STAINING_CELL_TYPES.get(position_type, position_type)
                if position_type is None:
                    continue  # Still none after replacements, skip
                cell_types_predicted[_get_cell_key(experiment, position)] = position_type

    data_frame = pandas.DataFrame.from_dict(cell_types_predicted, orient="index", columns=["immunostaining"])
    data_frame["immunostaining"] = pandas.Categorical(data_frame["immunostaining"])
    return data_frame


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
    experiments_immunostaining = list_io.load_experiment_list_file(_DATA_FILE_STAINING, load_images=False)
    staining_table = _get_staining_table(experiments_immunostaining)

    experiments_predictions = list_io.load_experiment_list_file(_DATA_FILE_PREDICTIONS, load_images=False)
    adata_predictions = _get_adata_predictions(experiments_predictions)
    adata_predictions = lib_figures.standard_preprocess(adata_predictions, log1p=False, scale=False)

    # Add the immunostaining to the predicted data, so that we have one dataset with all information
    combined = pandas.merge(adata_predictions.obs, staining_table, left_index=True, right_index=True,
                            how="left")
    adata_predictions.obs["immunostaining"] = combined["immunostaining"]

    # Plot the LDA
    figure = lib_figures.new_figure(size=(9, 6), dpi=600)
    conditions = list(_Condition)
    cell_types = adata_predictions.obs["immunostaining"].array.categories
    axes = numpy.array(figure.subplots(nrows=len(cell_types), ncols=len(conditions), sharex=True, sharey=True)).flatten()
    for i in range(len(conditions)):
        for j in range(len(cell_types)):
            is_last = i == len(conditions) - 1 and j == len(cell_types) - 1
            ax = axes[i + j * len(conditions)]
            condition = conditions[i]
            cell_type = cell_types[j]

            _plot_lda(ax, lda, adata, legend=is_last, colors=_desaturate(lib_figures.CELL_TYPE_PALETTE))

            # Highlight the cells that are of the current cell type
            cell_type_mask = ((adata_predictions.obs["immunostaining"] == cell_type)
                              & (adata_predictions.obs["treatment"] == condition.display_name))
            plot_coords = lda.transform(adata_predictions.X[cell_type_mask,])
            ax.scatter(plot_coords[:, 0], plot_coords[:, 1],
                       s=8, lw=0, c="black")

            ax.set_title(condition.display_name + " - " + cell_type)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_ylim(4, -8)
            ax.set_xlim(-4, 7)
            ax.set_aspect("equal")
    plt.show()


def _plot_lda(ax: Axes, lda: LinearDiscriminantAnalysis, adata: AnnData, *, legend: bool,
              colors: Dict[str, str] = lib_figures.CELL_TYPE_PALETTE, s: float = 8):
    plot_coords = lda.transform(adata.X)

    used_cell_types = adata.obs["cell_type_training"].array.categories
    colors = [colors[cell_type] for cell_type in adata.obs["cell_type_training"]]
    ax.scatter(plot_coords[:, 0], plot_coords[:, 1], s=s, lw=0, c=colors)

    if legend:
        # Add legend entry for each cell type
        for cell_type in used_cell_types:
            ax.scatter([], [], label=lib_figures.style_cell_type_name(cell_type), s=s, lw=0,
                       color=lib_figures.CELL_TYPE_PALETTE[cell_type])

        ax.legend(
            loc='center left', bbox_to_anchor=(1, 0.5)
        )


if __name__ == "__main__":
    main()
