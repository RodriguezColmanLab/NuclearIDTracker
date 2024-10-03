from typing import NamedTuple, Dict, Optional, List

import numpy
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from numpy import ndarray
import lib_figures
from organoid_tracker.core.position import Position
from organoid_tracker.imaging import list_io
from organoid_tracker.position_analysis import position_markers

_DATA_FILE_PREDICTED = "../../Data/Testing data - predictions - treatments - fixed.autlist"
_DATA_FILE_STAINING = "../../Data/Immunostaining conditions.autlist"
_TREATMENT_TRANSLATION = {
    "control": "Control",
    "dapt chir": "+DAPT +CHIR",
    "EN": "-Rspondin",
}
_PREDICTED_CELL_TYPES = {
    "ENTEROCYTE": "enterocyte",
    "PANETH": "Paneth",
    "STEM": "stem",
    "MATURE_GOBLET": None  # This type gets deleted, since we don't stain for it
}
_STAINING_CELL_TYPES = {
    None: "double-negative",
    "PANETH": "lysozym-positive",
    "ENTEROCYTE": "KRT20-positive"
}


class _ExperimentAndPosition(NamedTuple):
    """A combination of experiment name and position."""
    experiment_name: str
    position: Position

    def treatment(self) -> str:
        return self.experiment_name.split("-")[0]


class _ConfusionMatrix:
    """A confusion matrix with a fixed list of column names and row names."""
    predicted_names: List[str]
    actual_names: List[str]
    data: ndarray

    def __init__(self, predicted_names: List[str], actual_names: List[str]):
        self.predicted_names = predicted_names
        self.actual_names = actual_names
        self.data = numpy.zeros((len(actual_names), len(predicted_names)), dtype=int)

    def add(self, predicted: str, actual: str):
        """Add a single count to the confusion matrix."""
        predicted_index = self.predicted_names.index(predicted)
        actual_index = self.actual_names.index(actual)
        self.data[actual_index, predicted_index] += 1


def _get_confusion_matrix(cell_types_predicted: Dict[_ExperimentAndPosition, str],
                          cell_types_staining: Dict[_ExperimentAndPosition, str], treatment: str):
    """Get the confusion matrix for a specific treatment."""
    available_cell_types_predicted = [cell_type for cell_type in _PREDICTED_CELL_TYPES.values() if cell_type is not None]
    available_cell_types_staining = list(_STAINING_CELL_TYPES.values())

    confusion_matrix = _ConfusionMatrix(available_cell_types_predicted, available_cell_types_staining)
    for experiment_and_position, cell_type_predicted in cell_types_predicted.items():
        if experiment_and_position.treatment() != treatment:
            continue
        cell_type_staining = cell_types_staining.get(experiment_and_position)
        if cell_type_staining is not None:
            confusion_matrix.add(cell_type_predicted, cell_type_staining)

    return confusion_matrix


def _plot_confusion_matrix(confusion_matrix: _ConfusionMatrix, ax: Axes):
    # Scale the confusion matrix to percentages
    confusion_matrix_scaled = confusion_matrix.data
    for i in range(confusion_matrix_scaled.shape[0]):
        confusion_matrix_scaled[i, :] = confusion_matrix_scaled[i, :] / confusion_matrix_scaled[i, :].sum() * 100

    # Plot the confusion matrix
    ax.imshow(confusion_matrix_scaled, cmap="Blues", vmin=0, vmax=80)

    # Add percentages to the cells
    for i in range(confusion_matrix_scaled.shape[0]):
        for j in range(confusion_matrix_scaled.shape[1]):
            color = "black" if confusion_matrix_scaled[i, j] < 50 else "white"
            ax.text(j, i, f"{round(confusion_matrix_scaled[i, j])}%", ha="center", va="center", color=color)

    # Set the ticks and labels
    ax.set_xticks(numpy.arange(len(confusion_matrix.predicted_names)))
    ax.set_xticklabels(confusion_matrix.predicted_names, rotation=45, horizontalalignment="right")
    ax.set_yticks(numpy.arange(len(confusion_matrix.actual_names)))
    ax.set_yticklabels(confusion_matrix.actual_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Immunostaining")


def main():
    cell_types_predicted = _get_cell_types_dict(_DATA_FILE_PREDICTED, replacements=_PREDICTED_CELL_TYPES)
    cell_types_staining = _get_cell_types_dict(_DATA_FILE_STAINING, replacements=_STAINING_CELL_TYPES)

    treatments = list(_TREATMENT_TRANSLATION.keys())

    figure = lib_figures.new_figure()
    axes = numpy.array(figure.subplots((len(treatments) + 1) // 2, 2, sharex=True, sharey=True)).flatten()

    for ax, treatment in zip(axes, treatments):
        confusion_matrix = _get_confusion_matrix(cell_types_predicted, cell_types_staining, treatment)
        ax.set_title(_TREATMENT_TRANSLATION.get(treatment, treatment))
        _plot_confusion_matrix(confusion_matrix, ax)
    plt.show()


def _get_cell_types_dict(data_file: str, *, replacements: Dict[Optional[str], str] = None
                         ) -> Dict[_ExperimentAndPosition, str]:
    """Get a dictionary of cell types for each position in each experiment."""

    cell_types_predicted = dict()
    for experiment in list_io.load_experiment_list_file(data_file):
        print(f"Working on {experiment.name.get_name()}...")
        experiment_name = experiment.name.get_name()
        position_data = experiment.position_data

        for time_point in experiment.positions.time_points():
            for position in experiment.positions.of_time_point(time_point):
                position_type = position_markers.get_position_type(position_data, position)
                if replacements is not None:
                    position_type = replacements.get(position_type, position_type)
                if position_type is None:
                    continue  # Still none after replacements, skip
                cell_types_predicted[_ExperimentAndPosition(experiment_name, position)] = position_type
    return cell_types_predicted


if __name__ == "__main__":
    main()
