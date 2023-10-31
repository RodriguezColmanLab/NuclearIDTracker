import os
from collections import defaultdict
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy
import pandas
import scanpy
import scanpy.preprocessing
import sklearn.metrics
from matplotlib.axes import Axes
from numpy import ndarray
from sklearn.model_selection import KFold

from lib_models import ModelInputOutput, build_random_forest

_NUM_FOLDS = 5
_TREE_COUNTS = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]
_OUTPUT_FOLDER = "../Data/Models"
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
    accuracies: List[float]
    confusion_matrix: Optional[ndarray]

    def __init__(self, *, accuracy: Optional[float] = None, confusion_matrix: Optional[ndarray] = None):
        self.accuracies = list()
        if accuracy is not None:
            self.accuracies.append(accuracy)
        self.confusion_matrix = confusion_matrix

    def append(self, result: "_SingleParameterResults"):
        if result.confusion_matrix is not None:
            if self.confusion_matrix is None:
                self.confusion_matrix = result.confusion_matrix
            else:
                self.confusion_matrix += result.confusion_matrix
        self.accuracies += result.accuracies


class _ResultContainer:
    _accuracy_by_tree_count: Dict[int, _SingleParameterResults]

    def __init__(self):
        self._accuracy_by_tree_count = defaultdict(_SingleParameterResults)

    def post(self, tree_count: int, result: _SingleParameterResults):
        self._accuracy_by_tree_count[tree_count].append(result)

    def get_mean_accuracies(self, tree_counts_list: List[int]) -> ndarray:
        return numpy.array([numpy.mean(self._accuracy_by_tree_count[tree_count].accuracies)
                            for tree_count in tree_counts_list])

    def get_standard_deviations(self, tree_counts_list: List[int]) -> ndarray:
        return numpy.array([numpy.std(self._accuracy_by_tree_count[tree_count].accuracies, ddof=1)
                            for tree_count in tree_counts_list])

    def get_confusion_matrix(self, tree_count: int) -> ndarray:
        return self._accuracy_by_tree_count[tree_count].confusion_matrix


def main():
    results = _ResultContainer()

    adata = scanpy.read_h5ad(_TRAINING_DATA_FILE)
    adata = adata[adata.obs["cell_type_training"] != "NONE"]  # These ones have an unclear training cell type

    cell_types = adata.obs["cell_type_training"].array.categories
    input_output = ModelInputOutput(cell_type_mapping=list(cell_types), input_mapping=list(adata.var_names))
    x_values = adata.X
    y_values = numpy.array(adata.obs["cell_type_training"].cat.codes.array)

    for tree_count in _TREE_COUNTS:
        print(f"Working on tree count {tree_count}...")
        kfold = KFold(n_splits=_NUM_FOLDS, shuffle=True)
        weights_train = _calculate_class_weights(adata.obs["cell_type"])
        saved_a_model = False
        for train_indices, test_indices in kfold.split(x_values, y_values):
            # Build and train a neural network
            model = build_random_forest(input_output, tree_count=tree_count)
            model.fit(x_values[train_indices, :], y_values[train_indices],
                      class_weights=weights_train)

            # Evaluate the model
            scores = model.evaluate(x_values[test_indices], y_values[test_indices])
            confusion_matrix = numpy.array(sklearn.metrics.confusion_matrix(
                y_values[test_indices],
                numpy.argmax(model.predict(x_values[test_indices]), axis=1)))

            results.post(tree_count, _SingleParameterResults(
                accuracy=scores["accuracy"], confusion_matrix=confusion_matrix))

            # Save the first model (out of all cross-validated models)
            if not saved_a_model:
                folder = os.path.join(_OUTPUT_FOLDER, f"random-forest-with-{tree_count}-trees")
                os.makedirs(folder, exist_ok=True)
                model.save(folder)
                saved_a_model = True

    _plot_results(results)
    print(cell_types)
    print(results.get_confusion_matrix(100))


def _plot_results(results: _ResultContainer):
    figure = plt.figure()
    ax: Axes = figure.gca()
    color = "#0877cc"
    accuracy_means = results.get_mean_accuracies(_TREE_COUNTS)
    accuracy_stdevs = results.get_standard_deviations(_TREE_COUNTS)
    ax.plot(_TREE_COUNTS, accuracy_means, linewidth=2, markersize=12, color=color)
    ax.errorbar(_TREE_COUNTS, accuracy_means, yerr=accuracy_stdevs, color=color, capsize=2)
    ax.set_xscale("log")
    ax.set_ylim(0, 1)
    ax.set_xlabel("Number of trees")
    ax.set_ylabel("Accuracy (± st.dev.)")
    plt.show()


if __name__ == "__main__":
    main()
