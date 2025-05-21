from typing import Dict, List, Optional, Tuple, NamedTuple

import numpy
import pandas
import scanpy
import scanpy.preprocessing
import sklearn.metrics
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from numpy import ndarray
from sklearn.model_selection import KFold
from lib_models import build_shallow_model, ModelInputOutput
import lib_figures

_NUM_FOLDS = 5
_TRAINING_DATA_FILE = "../../Data/all_data.h5ad"


class _ConfusionMatrix(NamedTuple):
    cell_types: List[str]
    confusion_matrix: ndarray

    def accuracy(self) -> float:
        correct_predictions = numpy.diagonal(self.confusion_matrix).sum()
        total_predictions = self.confusion_matrix.sum()
        return correct_predictions / total_predictions if total_predictions > 0 else 0.0


def _calculate_class_weights(cell_types: pandas.Series) -> Dict[int, float]:
    counts_by_type = dict()

    array = cell_types.cat.codes.array
    for i in numpy.unique(array):
        count = numpy.count_nonzero(array == i)
        counts_by_type[i] = count

    max_count = max(counts_by_type.values())
    weights_by_type = dict()
    for i, count in counts_by_type.items():
        weights_by_type[i] = min(max_count / count, 2)

    return weights_by_type


class _SingleParameterResults:
    """Holds the results of a single parameter set, like for 20 hidden neurons and 5 epochs."""
    confusion_matrix: Optional[ndarray]

    def __init__(self, *, confusion_matrix: Optional[ndarray] = None):
        self.confusion_matrix = confusion_matrix

    def append(self, result: "_SingleParameterResults"):
        if result.confusion_matrix is not None:
            if self.confusion_matrix is None:
                self.confusion_matrix = result.confusion_matrix
            else:
                self.confusion_matrix += result.confusion_matrix


def main():
    comparison = _evaluate_model()
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


def _evaluate_model() -> _ConfusionMatrix:
    adata = scanpy.read_h5ad(_TRAINING_DATA_FILE)
    adata = adata[adata.obs["cell_type_training"] != "NONE"]  # These ones have an unclear training cell type
    cell_types = adata.obs["cell_type_training"].array.categories
    input_output = ModelInputOutput(cell_type_mapping=list(cell_types), input_mapping=list(adata.var_names))
    x_values = adata.X
    y_values = numpy.array(adata.obs["cell_type_training"].cat.codes.array)
    random = numpy.random.RandomState(1234)
    kfold = KFold(n_splits=_NUM_FOLDS, shuffle=True, random_state=random)
    weights_train = _calculate_class_weights(adata.obs["cell_type_training"])
    results = _SingleParameterResults()
    for train_indices, test_indices in kfold.split(x_values, y_values):
        # Build and train the model
        model = build_shallow_model(input_output, x_values[train_indices], hidden_neurons=0)
        model.fit(x_values[train_indices, :], y_values[train_indices],
                  class_weights=weights_train)

        # Evaluate the model
        predictions = model.predict(x_values[test_indices])
        answers = y_values[test_indices]

        confusion_matrix = sklearn.metrics.confusion_matrix(
            answers, numpy.argmax(predictions, axis=1), labels=numpy.arange(len(cell_types)))

        results.append(_SingleParameterResults(confusion_matrix=confusion_matrix))
    return _ConfusionMatrix(cell_types=cell_types, confusion_matrix=results.confusion_matrix)


if __name__ == "__main__":
    main()
