from typing import Dict, List, Optional, Tuple

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
    cell_types, results = _evaluate_model()
    confusion_matrix = results.confusion_matrix
    fraction_correct = numpy.diagonal(confusion_matrix).sum() / confusion_matrix.sum()
    space_for_bars = len(cell_types) + 0.75

    figure = lib_figures.new_figure()
    ax: Axes = figure.gca()
    for i, cell_type_immunostaining in enumerate(cell_types):
        x_values = [i * space_for_bars + j for j in range(len(cell_types))]
        y_values = [confusion_matrix[i, j] / numpy.sum(confusion_matrix[i]) * 100 for j in range(len(cell_types))]
        ax.bar(x_values, y_values, color=[lib_figures.CELL_TYPE_PALETTE[cell_types[j]] for j in range(len(cell_types))],
               width=1, align="edge")
        for x, y, cell_type in zip(x_values, y_values, cell_types):
            if cell_type == cell_type_immunostaining:
                ax.text(x + 0.5, y - 1, lib_figures.style_cell_type_name(cell_type), rotation=90, horizontalalignment="center", verticalalignment="top",
                        color="white")
                # Highlight the bar
                # ax.bar([x], [y], color=[(0, 0, 0, 0)], width=1, align="edge", linewidth=3, edgecolor="black")
            else:
                ax.text(x + 0.5, y + 3, lib_figures.style_cell_type_name(cell_type), rotation=90, horizontalalignment="center")

    ax.set_xticks([space_for_bars * i + 2 for i in range(len(cell_types))])
    ax.set_xticklabels([lib_figures.style_cell_type_name(name) for name in cell_types])
    ax.set_xlabel("Cell type from immunostaining")
    ax.set_ylabel("Predicted types (%)")
    ax.set_title(f"Accuracy: {fraction_correct * 100:.1f}%")
    ax.set_ylim(0, 100)

    plt.show()


def _evaluate_model() -> Tuple[List[str], _SingleParameterResults]:
    adata = scanpy.read_h5ad(_TRAINING_DATA_FILE)
    adata = adata[adata.obs["cell_type_training"] != "NONE"]  # These ones have an unclear training cell type
    cell_types = adata.obs["cell_type_training"].array.categories
    input_output = ModelInputOutput(cell_type_mapping=list(cell_types), input_mapping=list(adata.var_names))
    x_values = adata.X
    y_values = numpy.array(adata.obs["cell_type_training"].cat.codes.array)
    kfold = KFold(n_splits=_NUM_FOLDS, shuffle=True)
    weights_train = _calculate_class_weights(adata.obs["cell_type"])
    results = _SingleParameterResults()
    for train_indices, test_indices in kfold.split(x_values, y_values):
        # Build and train the model
        model = build_shallow_model(input_output, x_values[train_indices], hidden_neurons=0)
        model.fit(x_values[train_indices, :], y_values[train_indices],
                  class_weights=weights_train)

        # Evaluate the model
        predictions = model.predict(x_values[test_indices])
        answers = y_values[test_indices]

        # Only keep predictions where highest is more than 0.1 higher than previuos
        # predictions_sorted = numpy.partition(predictions, len(cell_types) - 2, axis=1)
        # keep_predictions = numpy.abs(predictions_sorted[:, -2] - predictions_sorted[:, -1]) > 0.10
        # predictions = predictions[keep_predictions]
        # answers = answers[keep_predictions]

        confusion_matrix = sklearn.metrics.confusion_matrix(
            answers, numpy.argmax(predictions, axis=1), labels=numpy.arange(len(cell_types)))

        results.append(_SingleParameterResults(confusion_matrix=confusion_matrix))
    return cell_types, results


if __name__ == "__main__":
    main()
