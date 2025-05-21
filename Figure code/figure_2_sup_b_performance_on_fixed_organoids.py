from typing import List

import matplotlib.colors
import numpy
from matplotlib import pyplot as plt
from matplotlib.axes import Axes

import lib_figures
from organoid_tracker.imaging import list_io
from organoid_tracker.position_analysis import position_markers

_DATA_FILE_OLFM4_STAINING_AND_PREDICTIONS = "../../Data/Fixed cells control/Olfm4 staining.autlist"

_DATA_FILE_KRT20_LYZ_PREDICTIONS = "../../Data/Testing data - predictions - treatments - fixed.autlist"
_DATA_FILE_KRT20_LYZ_STAINING = "../../Data/Immunostaining conditions.autlist"


class _ConfusionMatrix:
    cell_types: List[str]
    confusion_matrix: numpy.ndarray

    def __init__(self, cell_types: List[str]):
        self.cell_types = cell_types
        self.confusion_matrix = numpy.zeros((len(cell_types), len(cell_types)), dtype=int)

    def add(self, actual_cell_type: str, predicted_cell_type: str):
        if predicted_cell_type not in self.cell_types:
            return
        actual_index = self.cell_types.index(actual_cell_type)
        predicted_index = self.cell_types.index(predicted_cell_type)
        self.confusion_matrix[actual_index, predicted_index] += 1

    def accuracy(self) -> float:
        correct_predictions = numpy.diagonal(self.confusion_matrix).sum()
        total_predictions = self.confusion_matrix.sum()
        return correct_predictions / total_predictions if total_predictions > 0 else 0.0


def main():

    # Collect data for OLFM4 staining (here predictions and staining are in the same file)
    comparison = _ConfusionMatrix(cell_types=["ENTEROCYTE", "PANETH", "STEM"])
    for experiment in list_io.load_experiment_list_file(_DATA_FILE_OLFM4_STAINING_AND_PREDICTIONS, load_images=False):

        cell_types = experiment.global_data.get_data("ct_probabilities")
        if cell_types is None:
            raise ValueError(f"No cell type predictions found in the experiment {experiment.name}.")

        for position in experiment.positions:
            stained_type = position_markers.get_position_type(experiment.position_data, position)
            if stained_type != "STEM":
                continue
            predicted_scores = experiment.position_data.get_position_data(position, "ct_probabilities")
            if predicted_scores is None:
                continue
            predicted_cell_type = cell_types[numpy.argmax(predicted_scores)]
            comparison.add(stained_type, predicted_cell_type)

    # Collect data for other stainings (here predictions and staining are in different files)
    experiments_staining = list(list_io.load_experiment_list_file(_DATA_FILE_KRT20_LYZ_STAINING, load_images=False))
    experiments_predictions = list(list_io.load_experiment_list_file(_DATA_FILE_KRT20_LYZ_PREDICTIONS, load_images=False))
    for experiment_staining, experiment_predictions in zip(experiments_staining, experiments_predictions):
        if experiment_staining.name.get_name() != experiment_predictions.name.get_name():
            raise ValueError("The autlist files do not match: " + experiment_staining.name.get_name() +
                             " and " + experiment_predictions.name.get_name())
        if "control" not in experiment_staining.name.get_name():
            continue  # We don't use the treated organoids for this figure

        cell_types = experiment_predictions.global_data.get_data("ct_probabilities")
        if cell_types is None:
            raise ValueError(f"No cell type predictions found in the experiment {experiment_predictions.name}.")

        for position in experiment_staining.positions:
            stained_type = position_markers.get_position_type(experiment_staining.position_data, position)
            if stained_type is None:
                continue
            predicted_scores = experiment_predictions.position_data.get_position_data(position, "ct_probabilities")
            if predicted_scores is None:
                continue
            predicted_cell_type = cell_types[numpy.argmax(predicted_scores)]
            if stained_type == "ENTEROCYTE" or stained_type == "PANETH":
                comparison.add(stained_type, predicted_cell_type)

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
    ax.set_xticklabels([lib_figures.style_cell_type_name(cell_type) for cell_type in comparison.cell_types],
                       rotation=-45, ha="left")
    ax.set_yticklabels([lib_figures.style_cell_type_name(cell_type) for cell_type in comparison.cell_types])
    ax.set_xlabel("Predicted cell type")
    ax.set_ylabel("Cell type (immunostaining)")

    # Add counts to all cells
    for i in range(len(comparison.cell_types)):
        for j in range(len(comparison.cell_types)):
            count = comparison.confusion_matrix[i, j]
            color = "white" if confusion_matrix_scaled[i, j] > 0.5 else "black"
            ax.text(j, i, str(count), ha="center", va="center", color=color)

    figure.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
