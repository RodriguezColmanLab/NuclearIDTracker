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
    space_for_bars = 3

    figure = lib_figures.new_figure()
    ax: Axes = figure.gca()
    for i, cell_type_immunostaining in enumerate(cell_types):
        true_positives = confusion_matrix[i, i]
        false_negatives = confusion_matrix[:, i].sum() - true_positives
        y_values = [true_positives, false_negatives]
        y_values = [value / sum(y_values) * 100 for value in y_values]
        ax.bar(i * space_for_bars + 1, y_values[0], color=lib_figures.CELL_TYPE_PALETTE["STEM"], label="True positives" if i == 0 else None)
        ax.text(i * space_for_bars + 1, y_values[0] + 1, str(true_positives), ha="center", va="bottom")
        ax.bar(i * space_for_bars + 2, y_values[1], color="#8e44ad", label="False negatives" if i == 0 else None)
        ax.text(i * space_for_bars + 2, y_values[1] + 1, str(false_negatives), ha="center", va="bottom")

    ax.set_xticks([space_for_bars * i + 1.5 for i in range(len(cell_types))])
    ax.set_xticklabels([lib_figures.style_cell_type_name(name) for name in cell_types])
    ax.set_xlabel("Cell type from immunostaining")
    ax.set_ylabel("Cells (%)")
    ax.set_title(f"Accuracy: {fraction_correct * 100:.1f}%")
    ax.set_ylim(0, 100)
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    figure.tight_layout()
    plt.show()


def _evaluate_model() -> Tuple[List[str], _SingleParameterResults]:
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
    return cell_types, results


if __name__ == "__main__":
    main()
