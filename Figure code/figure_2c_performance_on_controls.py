from typing import List

import numpy
from matplotlib import pyplot as plt
from matplotlib.axes import Axes

from organoid_tracker.core import TimePoint
from organoid_tracker.core.experiment import Experiment
from organoid_tracker.imaging import list_io
from organoid_tracker.position_analysis import position_markers

import lib_data
import lib_figures

_DATA_FILE_STAINING = "../../Data/Training data.autlist"
_DATA_FILE_PREDICTED = "../../Data/Tracking data as controls/Dataset - full overweekend.autlist"


class _ConfusionMatrix:
    cell_types: List[str]
    confusion_matrix: numpy.ndarray

    def __init__(self, cell_types: List[str]):
        self.cell_types = cell_types
        self.confusion_matrix = numpy.zeros((len(cell_types), len(cell_types)), dtype=int)

    def add(self, actual_cell_type: str, predicted_cell_type: str):
        actual_index = self.cell_types.index(actual_cell_type)
        predicted_index = self.cell_types.index(predicted_cell_type)
        self.confusion_matrix[actual_index, predicted_index] += 1

    def accuracy(self) -> float:
        correct_predictions = numpy.diagonal(self.confusion_matrix).sum()
        total_predictions = self.confusion_matrix.sum()
        return correct_predictions / total_predictions if total_predictions > 0 else 0.0


def _compare_experiments(experiment_stained: Experiment, experiment_predicted: Experiment, *, into: _ConfusionMatrix):
    last_time_point_number = experiment_stained.positions.last_time_point_number()
    time_points = [TimePoint(last_time_point_number), TimePoint(last_time_point_number - 5),
                   TimePoint(last_time_point_number - 10), TimePoint(last_time_point_number - 15)]

    for time_point in time_points:
        for position in experiment_stained.positions.of_time_point(time_point):
            position_type_stained = position_markers.get_position_type(experiment_stained.position_data, position)
            if position_type_stained is None:
                continue
            cell_type_stained = lib_data.convert_cell_type(position_type_stained)
            if cell_type_stained == "NONE":
                continue
            position_type_predicted = position_markers.get_position_type(experiment_predicted.position_data, position)
            if position_type_predicted is None:
                continue
            into.add(cell_type_stained, position_type_predicted)


def main():
    comparison = _calculate_confusion_matrix()
    _show_bar_plot(comparison)
    _show_confusion_matrix(comparison)


def _show_bar_plot(comparison: _ConfusionMatrix):
    cell_types = comparison.cell_types
    confusion_matrix = comparison.confusion_matrix
    fraction_correct = numpy.diagonal(confusion_matrix).sum() / confusion_matrix.sum()

    space_for_bars = 2.75

    figure = lib_figures.new_figure()
    ax: Axes = figure.gca()
    for i, cell_type_immunostaining in enumerate(cell_types):
        amount_correct_of_type = confusion_matrix[i, i]
        amount_incorrect_of_type = numpy.sum(confusion_matrix[i]) - amount_correct_of_type
        percentage_correct_of_type = confusion_matrix[i, i] / numpy.sum(confusion_matrix[i]) * 100
        x_values = [i * space_for_bars + j for j in [0, 1]]
        y_values = [percentage_correct_of_type, 100 - percentage_correct_of_type]
        ax.bar(x_values, y_values, color=["#85ff8e", "#a214dd"],
               width=1, align="edge")
        ax.text(x_values[0] + 0.5, y_values[0] - 1, str(amount_correct_of_type), ha="center", va="top")
        ax.text(x_values[1] + 0.5, y_values[1] + 1, str(amount_incorrect_of_type), ha="center", va="bottom")

    ax.set_xticks([space_for_bars * i + 1 for i in range(len(cell_types))])
    ax.set_xticklabels([lib_figures.style_cell_type_name(name) for name in cell_types])
    ax.set_xlabel("Cell type from immunostaining")
    ax.set_ylabel("Predicted types (%)")
    ax.set_title(f"Accuracy: {fraction_correct * 100:.1f}%")
    ax.set_ylim(0, 100)

    plt.show()


def _show_confusion_matrix(comparison: _ConfusionMatrix):
    figure = lib_figures.new_figure()
    ax = figure.gca()

    ax.set_title(f"Accuracy: {comparison.accuracy() * 100:.2f}%")
    confusion_matrix_scaled = comparison.confusion_matrix.astype(float)
    confusion_matrix_scaled /= confusion_matrix_scaled.sum(axis=1, keepdims=True)
    ax.imshow(confusion_matrix_scaled, cmap="Blues")
    ax.set_xticks(range(len(comparison.cell_types)))
    ax.set_yticks(range(len(comparison.cell_types)))
    ax.set_xticklabels([lib_figures.style_cell_type_name(cell_type) for cell_type in comparison.cell_types], rotation=-45, ha="left")
    ax.set_yticklabels([lib_figures.style_cell_type_name(cell_type) for cell_type in comparison.cell_types])
    ax.set_xlabel("Predicted cell type")
    ax.set_ylabel("Cell type (immunostaining)")

    # Add counts to all cells
    for i in range(len(comparison.cell_types)):
        for j in range(len(comparison.cell_types)):
            count = comparison.confusion_matrix[i, j]
            color = "white" if count > 75 else "black"
            ax.text(j, i, str(count), ha="center", va="center", color=color)

    figure.tight_layout()
    plt.show()


def _calculate_confusion_matrix() -> _ConfusionMatrix:
    print("Loading staining data...")
    experiments_stained = list(list_io.load_experiment_list_file(_DATA_FILE_STAINING, load_images=False))
    print("Loading predicted data...")
    confusion_matrix = None
    for experiment_predicted in list_io.load_experiment_list_file(_DATA_FILE_PREDICTED, load_images=False):
        if confusion_matrix is None:
            cell_types = experiment_predicted.global_data.get_data("ct_probabilities")
            if cell_types is None:
                raise ValueError("Cell types not found in predicted data")
            cell_types.remove("MATURE_GOBLET")  # Not used in these files
            confusion_matrix = _ConfusionMatrix(cell_types)

        for experiment_stained in experiments_stained:
            if experiment_stained.name == experiment_predicted.name:
                print(f"Comparing {experiment_stained.name}...")
                _compare_experiments(experiment_stained, experiment_predicted, into=confusion_matrix)
                break
    return confusion_matrix


if __name__ == "__main__":
    main()
